import os
import hydra
import torch
import torch.nn as nn
from hydra.utils import to_absolute_path
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
from contextlib import nullcontext
from typing import Tuple
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from data.build import get_dataset, get_split_dataloaders
from logs.logger import RunLogger, get_model_checkpoint
from logs.metrics import EpochMetric, SimpleMetric
from model.network import SynthRL
from model.loss import QuantizedNumericalParamsLoss, CategoricalParamsAccuracy, PresetProcessor, ParameterLoss, calculate_rewards
from utils.audio import AudioRenderer, Spectrogram_Processor
from utils.scheduler import linear_scheduler
from utils.hparams import LinearDynamicParam
from utils.distrib import get_parallel_devices
from utils.buffer import Replaybuffer


def load_source_cfg(cfg: DictConfig) -> Tuple[Path, DictConfig]:
    logs_root_dir = Path(to_absolute_path(cfg.logs_root_dir))
    src_run_dir = logs_root_dir / cfg.source_run_name
    src_cfg_path = src_run_dir / 'config.yaml'

    if not src_cfg_path.exists():
        raise FileNotFoundError(f'Saved config not found: {src_cfg_path}')
    
    src_cfg = OmegaConf.load(src_cfg_path)
    return src_run_dir, src_cfg


def finetune(cfg: DictConfig):
    src_run_dir, src_cfg = load_source_cfg(cfg)
    device, device_ids = get_parallel_devices(cfg.main_cuda_device_idx)
    
    synth_cfg = OmegaConf.load(f'config/dataset/{cfg.synth_name}.yaml')
    synth_dataset = get_dataset(synth_cfg)
    ft_dataset = get_dataset(cfg.dataset)
    dataloader = get_split_dataloaders(src_cfg.train, ft_dataset)
    preset_idx_helper = synth_dataset.preset_indexes_helper

    logger = RunLogger(cfg)

    model = SynthRL(preset_idx_helper, **src_cfg.model)
    model = model.to(device).train()
    checkpoint = get_model_checkpoint(src_run_dir, cfg.train.start_epoch, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model_parallel = nn.DataParallel(model, device_ids, device)

    # Parameter loss
    if cfg.loss.param:
        param_criterion = ParameterLoss(
            preset_idx_helper,
            cat_softmax_t=cfg.loss.cat_softmax_t,
            label_smoothing=cfg.loss.label_smoothing,
        )

    # RL loss
    if cfg.loss.rl:
        preset_processor = PresetProcessor(synth_dataset, preset_idx_helper, device)
        sample_rate = src_cfg.dataset.sample_rate

        audio_renderer = AudioRenderer(
            synth_dataset,
            sample_rate,
            cfg.loss.synth_render_workers,
            device,
        )

        spec_processor = Spectrogram_Processor(
            src_cfg.dataset.n_fft,
            src_cfg.dataset.fft_hop,
            sample_rate,
        ).to(device)

    # Monitoring error
    param_num_criterion = QuantizedNumericalParamsLoss(preset_idx_helper)
    param_cat_criterion = CategoricalParamsAccuracy(preset_idx_helper)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optim.initial_lr,
            weight_decay=cfg.optim.weight_decay,
            betas=cfg.optim.betas,
        )
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, eta_min=0.000001)
    
    # Replay buffer
    buffer = Replaybuffer(
        len(ft_dataset),
        preset_idx_helper.full_preset_size,
        cfg.loss.per_capacity,
        device
    )

    # Logger
    scalars_train, scalars_valid = {}, {}

    if cfg.loss.rl:
        scalars_train['RL/LogProb/Train'] = EpochMetric()
        scalars_train['RL/BackpropLoss/Train'] = EpochMetric()
        scalars_train['RL/Rewards/Train'] = EpochMetric()
        scalars_valid['RL/Rewards/Valid'] = EpochMetric()
        scalars_train['Metric/SC/Train'] = EpochMetric()
        scalars_train['Metric/SpecMAE/Train'] = EpochMetric()
        scalars_train['Metric/mfccMAE/Train'] = EpochMetric()
        scalars_valid['Metric/SC/Valid'] = EpochMetric()
        scalars_valid['Metric/SpecMAE/Valid'] = EpochMetric()
        scalars_valid['Metric/mfccMAE/Valid'] = EpochMetric()

    if cfg.loss.param:
        scalars_train['Params/BackpropLoss/Train'] = EpochMetric()
        scalars_valid['Params/BackpropLoss/Valid'] = EpochMetric()

    if cfg.dataset.name == 'dexed':
        scalars_train['Params/CatAcc/Train'] = EpochMetric()
        scalars_train['Params/NumMAE/Train'] = EpochMetric()
        scalars_valid['Params/CatAcc/Valid'] = EpochMetric()
        scalars_valid['Params/NumMAE/Valid'] = EpochMetric()
        
    scalars_train['Sched/LR'] = SimpleMetric(cfg.optim.initial_lr)
    scalars_train['Sched/LRwarmup'] = LinearDynamicParam(
        start_value=cfg.scheduler.warmup_start_factor,
        end_value=1.0,
        end_epoch=cfg.scheduler.warmup_epochs,
        current_epoch=0,
    )

    # Virtual display for the synthesizer
    try:
        from pyvirtualdisplay import Display
        disp = Display() if not os.environ.get('DISPLAY') else nullcontext()
    except Exception:
        disp = nullcontext()

    # Buffer initialization
    with disp:
        with torch.inference_mode():
            deterministic = True

            for epoch in tqdm(range(cfg.loss.per_capacity), desc='buffer init epoch', position=0):
                dataloader_iter = iter(dataloader['train'])

                for _ in tqdm(range(len(dataloader['train'])), desc='buffer init batch', position=1, leave=False):
                    sample = next(dataloader_iter)
                    x_wav = sample[0].to(device)
                    x_in = sample[1].to(device)
                    v_in = sample[2].to(device)
                    preset_UID = sample[4].to(device)
                    
                    v_out = model_parallel(x_in)
                    full_preset_out, actions = preset_processor(v_out, deterministic)
                    inferred_wavs = audio_renderer.multi_process_render(full_preset_out)
                    inferred_wavs = inferred_wavs / torch.abs(inferred_wavs + 1e-5).max(dim=1)[0].unsqueeze(1)
                    sc, log_mae, mfcc_mae, _ = spec_processor.calculate_metrics(x_wav, inferred_wavs)
                    rewards = calculate_rewards(sc, log_mae, mfcc_mae)
                    buffer.store(preset_UID, actions, rewards)

                deterministic = False

        buffer_path = logger.log_dir / 'buffer' / cfg.dataset.name
        buffer_path.mkdir(parents=True, exist_ok=True)
        buffer.save(buffer_path)        

        # Model training epochs
        for epoch in tqdm(range(cfg.train.start_epoch, cfg.train.end_epoch), desc='epoch', position=0):
            # Initialize logger metrics
            for s in scalars_train.values():
                s.on_new_epoch()

            # Learning rate warmup
            lr_epoch = epoch - cfg.train.start_epoch
            if lr_epoch <= cfg.scheduler.warmup_epochs:
                warm = scalars_train['Sched/LRwarmup'].get(lr_epoch)
                for g in optimizer.param_groups:
                    g['lr'] = warm * cfg.optim.initial_lr

            # Train
            model_parallel.train()
            dataloader_iter = iter(dataloader['train'])
            pbar = tqdm(range(len(dataloader['train'])), desc='train batch', position=1, leave=False)

            for _ in pbar:
                sample = next(dataloader_iter)
                x_wav = sample[0].to(device)
                x_in = sample[1].to(device)
                v_in = sample[2].to(device)
                preset_UID = sample[4].to(device)
                optimizer.zero_grad()
                v_out = model_parallel(x_in)

                if cfg.loss.rl:
                    full_preset_out, actions = preset_processor(v_out, deterministic=False)
                    inferred_wavs = audio_renderer.multi_process_render(full_preset_out)
                    inferred_wavs = inferred_wavs / torch.abs(inferred_wavs + 1e-5).max(dim=1)[0].unsqueeze(1)
                    sc, log_mae, mfcc_mae, _ = spec_processor.calculate_metrics(x_wav, inferred_wavs)
                    rewards = calculate_rewards(sc, log_mae, mfcc_mae)

                    if cfg.loss.per:
                        buffer.store(preset_UID, actions, rewards)
                        actions, rewards = buffer.sample(preset_UID)
                        
                    mean_log_probs = preset_processor.get_mean_log_probs(v_out, actions)
                    rl_loss = -(cfg.loss.per_capacity * rewards * mean_log_probs).mean()
                    
                    c = cfg.loss.rl_coef
                    alpha = linear_scheduler(epoch, c['s_value'], c['e_value'], c['s_epoch'], c['e_epoch'])
                    
                    scalars_train['RL/LogProb/Train'].append(mean_log_probs.mean().item())
                    scalars_train['RL/BackpropLoss/Train'].append(rl_loss.item())
                    scalars_train['RL/Rewards/Train'].append(rewards.mean().item())
                    scalars_train['Metric/SC/Train'].append(sc.mean().item())
                    scalars_train['Metric/SpecMAE/Train'].append(log_mae.mean().item())
                    scalars_train['Metric/mfccMAE/Train'].append(mfcc_mae.mean().item())
                else:
                    rl_loss = torch.zeros(1).to(device)
                    alpha = 0

                if cfg.loss.param:
                    param_loss = param_criterion(v_out, v_in)
                    scalars_train['Params/BackpropLoss/Train'].append(param_loss.item())
                else:
                    param_loss = torch.zeros(1, device=device)
                    alpha = 1

                if cfg.dataset.name == 'dexed':
                    with torch.no_grad():
                        scalars_train['Params/NumMAE/Train'].append(param_num_criterion(v_out, v_in))
                        scalars_train['Params/CatAcc/Train'].append(param_cat_criterion(v_out, v_in))
                
                loss = alpha * rl_loss + (1 - alpha) * param_loss
                loss.backward()
                optimizer.step()

            scalars_train['Sched/LR'] = SimpleMetric(optimizer.param_groups[0]['lr'])
            scheduler.step()

            for k, s in scalars_train.items():
                logger.tensorboard.add_scalar(k, s.get(), epoch)

            # Evaluation on validation dataset
            if (epoch % cfg.train.eval_period == 0) or (epoch == cfg.train.end_epoch - 1):
                for s in scalars_valid.values():
                    s.on_new_epoch()
                
                model_parallel.eval()
                pbar = tqdm(dataloader['validation'], desc='val batch', position=1, leave=False)

                with torch.inference_mode():    
                    for sample in pbar:
                        x_wav = sample[0].to(device)
                        x_in = sample[1].to(device)
                        v_in = sample[2].to(device)
                        v_out = model_parallel(x_in)

                        if cfg.loss.rl:
                            full_preset_out, actions = preset_processor(v_out, deterministic=True)
                            inferred_wavs = audio_renderer.multi_process_render(full_preset_out)
                            inferred_wavs = inferred_wavs / torch.abs(inferred_wavs + 1e-5).max(dim=1)[0].unsqueeze(1)
                            sc, log_mae, mfcc_mae, _ = spec_processor.calculate_metrics(x_wav, inferred_wavs)
                            rewards = calculate_rewards(sc, log_mae, mfcc_mae)
                            scalars_valid['Metric/SC/Valid'].append(sc.mean().item())
                            scalars_valid['Metric/SpecMAE/Valid'].append(log_mae.mean().item())
                            scalars_valid['Metric/mfccMAE/Valid'].append(mfcc_mae.mean().item())
                            scalars_valid['RL/Rewards/Valid'].append(rewards.mean().item())

                        if cfg.loss.param:
                            param_loss = param_criterion(v_out, v_in)
                            scalars_valid['Params/BackpropLoss/Valid'].append(param_loss.item())
                        
                    if cfg.dataset.name == 'dexed':
                        # Monitoring loss
                        scalars_valid['Params/NumMAE/Valid'].append(param_num_criterion(v_out, v_in))
                        scalars_valid['Params/CatAcc/Valid'].append(param_cat_criterion(v_out, v_in))

                for k, s in scalars_valid.items():
                    logger.tensorboard.add_scalar(k, s.get(), epoch)
            
            # Save checkpoint
            if (epoch > 0 and epoch % cfg.train.save_period == 0) or (epoch == cfg.train.end_epoch - 1):
                logger.save_checkpoint(epoch, model, optimizer, scheduler)
            
            logger.on_epoch_finished(epoch)

    logger.on_training_finished()
    print('Finetuning process finished')


@hydra.main(config_path='config', version_base='1.3')
def main(cfg: DictConfig):
    finetune(cfg)


if __name__ == '__main__':
    main()
