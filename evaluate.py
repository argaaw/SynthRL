import os
import hydra
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
from contextlib import nullcontext
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig


from model.network import SynthRL
from utils.audio import AudioRenderer, Spectrogram_Processor
from data.build import get_dataset, get_split_dataloaders
from logs.logger import get_model_checkpoint
from model.loss import PresetProcessor, QuantizedNumericalParamsLoss, CategoricalParamsAccuracy


def evaluate(cfg: DictConfig):
    logs_root_dir = Path(to_absolute_path(cfg.logs_root_dir))
    log_dir = logs_root_dir / cfg.run_name
    train_cfg = OmegaConf.load(log_dir / 'config.yaml')
    train_cfg.train.minibatch_size = cfg.minibatch_size

    eval_path = log_dir / cfg.dataset.name / f'{cfg.ckpt_epoch:03d}epoch'
    audio_path = eval_path / 'audio'
    audio_path.mkdir(parents=True, exist_ok=True)
    eval_pkl_path = eval_path / 'eval.pickle'

    synth_cfg = OmegaConf.load(f'config/dataset/{cfg.synth_name}.yaml')
    synth_dataset = get_dataset(synth_cfg)
    eval_dataset = get_dataset(cfg.dataset)
    dataloader = get_split_dataloaders(train_cfg.train, eval_dataset)
    dataloader = dataloader[cfg.split]
    preset_idx_helper = synth_dataset.preset_indexes_helper

    device = torch.device(cfg.device)
    ckpt = get_model_checkpoint(log_dir, cfg.ckpt_epoch, device)
    model = SynthRL(preset_idx_helper, **train_cfg.model).to(device).eval()
    torch.set_grad_enabled(False)
    model.load_state_dict(ckpt['model_state_dict'])

    param_num_criterion = QuantizedNumericalParamsLoss(
        preset_idx_helper,
        nn.L1Loss(),
        reduce=False,
    )

    param_cat_criterion = CategoricalParamsAccuracy(
        preset_idx_helper,
        reduce=False,
        percentage_output=True,
    )

    sample_rate = train_cfg.dataset.sample_rate
    preset_processor = PresetProcessor(synth_dataset, preset_idx_helper, device)

    audio_renderer = AudioRenderer(
        synth_dataset,
        sample_rate,
        num_workers=1,
        device=device,
    )

    spec_processor = Spectrogram_Processor(
        train_cfg.dataset.n_fft,
        train_cfg.dataset.fft_hop,
        sample_rate,
    ).to(device)

    eval_metrics = []
    assert cfg.minibatch_size == 1

    try:
        from pyvirtualdisplay import Display
        disp = Display() if not os.environ.get('DISPLAY') else nullcontext()
    except Exception:
        disp = nullcontext()

    with disp, torch.inference_mode():
        for sample in tqdm(dataloader):
            x_wav, x_in, v_in, preset_UID, _ = sample
            x_wav = x_wav.to(device)
            x_in = x_in.to(device)
            v_in = v_in.to(device)
            preset_UID = preset_UID.item()           

            v_out = model(x_in)
            full_preset_out, _ = preset_processor(v_out, deterministic=True)
            inferred_wav = audio_renderer.single_process_render(full_preset_out)
            inferred_wav = inferred_wav / torch.abs(inferred_wav + 1e-5).max(1)[0]
            metric_values = spec_processor.calculate_metrics(x_wav, inferred_wav)
            sc, log_mae, mfcc13_mae, mfcc40_mae = metric_values

            filename_gt = audio_path.joinpath(f'{preset_UID}_gt.wav')
            filename_inferred = audio_path.joinpath(f'{preset_UID}.wav')
            sf.write(filename_gt, x_wav.cpu().numpy()[0], sample_rate)
            sf.write(filename_inferred, inferred_wav.cpu().numpy()[0], sample_rate)
            
            row = {
                'preset_UID': preset_UID,
                'spec_sc': sc.item(),
                'spec_mae': log_mae.item(),
                'mfcc13_mae': mfcc13_mae.item(),
                'mfcc40_mae': mfcc40_mae.item(),
            }

            if cfg.dataset.name == 'dexed':
                accuracies = param_cat_criterion(v_out, v_in)
                acc_value = np.asarray([v for _, v in accuracies.items()]).mean()
                _, mae_value = param_num_criterion(v_out, v_in)
                row['num_mae'] = mae_value.item()
                row['cat_acc'] = acc_value

            eval_metrics.append(row)

    eval_df = pd.DataFrame(eval_metrics)
    eval_df = eval_df.groupby('preset_UID', as_index=False).mean(numeric_only=True)
    eval_df.to_pickle(eval_pkl_path)
    print('Finished evaluation')
        

@hydra.main(config_path="config", config_name="eval", version_base="1.3")
def main(cfg: DictConfig):
    evaluate(cfg)


if __name__ == "__main__":
    main()
