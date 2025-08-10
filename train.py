import hydra
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from omegaconf import DictConfig
from tqdm import tqdm

from data.build import get_dataset, get_split_dataloaders
from logs.logger import RunLogger, get_model_checkpoint
from logs.metrics import SimpleMetric, EpochMetric
from model.network import SynthRL
from model.loss import ParameterLoss, QuantizedNumericalParamsLoss, CategoricalParamsAccuracy
from utils.distrib import get_parallel_devices
from utils.hparams import LinearDynamicParam


def train(cfg: DictConfig):
    # Prepare dataloader and synthesizer
    dataset = get_dataset(cfg.dataset)
    dataloader = get_split_dataloaders(cfg.train, dataset)
    logger = RunLogger(cfg)
    preset_idx_helper = dataset.preset_indexes_helper

    # Initialize the model
    model = SynthRL(preset_idx_helper, **cfg.model)
    device, device_ids = get_parallel_devices(cfg.main_cuda_device_idx)
    model = model.to(device)
    model_parallel = torch.nn.DataParallel(model, device_ids, device)

    # Parameter loss
    param_criterion = ParameterLoss(
        preset_idx_helper,
        cat_softmax_t=cfg.loss.cat_softmax_t,
        label_smoothing=cfg.loss.label_smoothing,
    )

    # Monitoring error
    params_num_criterion = QuantizedNumericalParamsLoss(preset_idx_helper)
    params_cat_criterion = CategoricalParamsAccuracy(preset_idx_helper)

    # Tensorboard
    scalars_train = dict()
    scalars_valid = dict()
    scalars_train['Params/BackpropLoss/Train'] = EpochMetric()
    scalars_valid['Params/BackpropLoss/Valid'] = EpochMetric()
    scalars_train['Params/NumMAE/Train'] = EpochMetric()
    scalars_valid['Params/NumMAE/Valid'] = EpochMetric()
    scalars_train['Params/CatAcc/Train'] = EpochMetric()
    scalars_valid['Params/CatAcc/Valid'] = EpochMetric()

    scalars_train['Sched/LR'] = SimpleMetric(cfg.optim.initial_lr)
    scalars_train['Sched/LRwarmup'] = LinearDynamicParam(
        start_value=cfg.scheduler.warmup_start_factor,
        end_value=1.0,
        end_epoch=cfg.scheduler.warmup_epochs,
        current_epoch=cfg.train.start_epoch
    )

    # Optimizer and Scheduler
    model.train()
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optim.initial_lr,
        weight_decay=cfg.optim.weight_decay,
        betas=cfg.optim.adam_betas,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, eta_min=0.000001)

    if cfg.train.start_epoch > 0:
        checkpoint = get_model_checkpoint(logger.checkpoints_dir, cfg.train.start_epoch, device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Model training epochs
    for epoch in tqdm(range(cfg.train.start_epoch, cfg.train.end_epoch), desc='epoch', position=0):
        # Initialize logger metrics
        for _, s in scalars_train.items():
            s.on_new_epoch()

        # Learning rate warmup
        if epoch <= cfg.scheduler.warmup_epochs:
            warm = scalars_train['Sched/LRwarmup'].get(epoch)
            for g in optimizer.param_groups:
                g['lr'] = warm * cfg.optim.initial_lr

        # Train
        model_parallel.train()
        dataloader_iter = iter(dataloader['train'])
        pbar = tqdm(range(len(dataloader['train'])), desc='train batch', position=1, leave=False)
        
        for _ in pbar:
            sample = next(dataloader_iter)
            x_in = sample[1].to(device)
            v_in = sample[2].to(device)
            optimizer.zero_grad()
            v_out = model_parallel(x_in)

            # Monitoring error
            with torch.no_grad():
                scalars_train['Params/NumMAE/Train'].append(params_num_criterion(v_out, v_in))
                scalars_train['Params/CatAcc/Train'].append(params_cat_criterion(v_out, v_in))
            
            param_loss = param_criterion(v_out, v_in)

            # Parameter loss
            scalars_train['Params/BackpropLoss/Train'].append(param_loss)
            param_loss.backward()
            optimizer.step()

        # Evaluation on validation dataset
        if (epoch % cfg.train.eval_period == 0) or (epoch == cfg.train.end_epoch - 1):
            for s in scalars_valid.values():
                s.on_new_epoch()

            model_parallel.eval()
            pbar = tqdm(dataloader['validation'], desc='val batch', poisition=1, leave=False)

            with torch.inference_mode():
                for sample in pbar:
                    x_in = sample[1].to(device)
                    v_in = sample[2].to(device)
                    v_out = model_parallel(x_in)

                    # Parameter loss
                    param_loss = param_criterion(v_out, v_in)
                    scalars_valid['Params/BackpropLoss/Valid'].append(param_loss)
                    
                    # Monitoring error
                    scalars_valid['Params/NumMAE/Valid'].append(params_num_criterion(v_out, v_in))
                    scalars_valid['Params/CatAcc/Valid'].append(params_cat_criterion(v_out, v_in))

            for k, s in scalars_valid.items():
                logger.tensorboard.add_scalar(k, s.get(), epoch)

        scalars_train['Sched/LR'] = SimpleMetric(optimizer.param_groups[0]['lr'])
        scheduler.step()

        # Logging
        for k, s in scalars_train.items():
            logger.tensorboard.add_scalar(k, s.get(), epoch)

        # Save checkpoint
        if (epoch > 0 and epoch % cfg.train.save_period == 0) or (epoch == cfg.train.end_epoch - 1):
            logger.save_checkpoint(epoch, model, optimizer, scheduler)

        logger.on_epoch_finished(epoch)

    logger.on_training_finished()
    print('Training finished')


@hydra.main(config_path='config', config_name='stage1', version_base='1.3')
def main(cfg: DictConfig):
    train(cfg)


if __name__ == '__main__':
    main()
