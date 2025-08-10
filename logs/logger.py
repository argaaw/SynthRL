import datetime
import humanize
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

from logs.tbwriter import TensorboardSummaryWriter  # Custom modified summary writer


def get_model_checkpoint(log_dir: Path, epoch, device=None):
    """ Returns the path to a .tar saved checkpoint, or prints all available checkpoints and raises an exception
    if the required epoch has no saved .tar checkpoint. """
    checkpoint_path = log_dir / f'checkpoints/{epoch:05d}.tar'

    try:
        if device is None:
            checkpoint = torch.load(checkpoint_path)  # Load on original device
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)  # e.g. train on GPU, load on CPU
    except (OSError, IOError) as e:
        available_checkpoints = "Available checkpoints: {}".format([f.name for f in log_dir.glob('*.tar')])
        print(available_checkpoints)
        raise ValueError("Cannot load checkpoint for epoch {}: {}".format(epoch, e))
    
    return checkpoint


def get_model_last_checkpoint(root_path: Path, config, verbose=True, device=None):
    checkpoints_dir = root_path.joinpath(config.model.name, config.model.run_name, 'checkpoints')
    available_epochs = [int(f.stem) for f in checkpoints_dir.glob('*.tar')]
    assert len(available_epochs) > 0  # At least 1 checkpoint should be available
    if verbose:
        print("Loading epoch {} from {}".format(max(available_epochs), checkpoints_dir))
    return get_model_checkpoint(root_path, config, max(available_epochs), device)


class RunLogger:
    """ Class for saving interesting data during a training run:
     - graphs, losses, metrics, and some results to Tensorboard
     - config.py as a json file
     - trained models
     """
    def __init__(self, config):
        """

        :param root_path: Path of the project's root folder
        :param model_config: from config.py
        :param train_config: from config.py
        :param minibatches_count: Length of the 'train' dataloader
        """
        # Configs are stored but not modified by this class
        self.config = config
        self.verbosity = config.verbosity

        # Directories creation (if not exists) for model
        self.log_dir = Path(config.logs_root_dir) / config.run_name
        self.checkpoints_dir = self.log_dir / 'checkpoints'
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        if self.verbosity >= 1:
            print(f"[RunLogger] Starting logging into '{str(self.log_dir)}'")

        self.epoch_start_datetimes = [datetime.datetime.now()]
        
        # Tensorboard
        self.tensorboard = TensorboardSummaryWriter(
            log_dir=self.log_dir,
            flush_secs=5,
            config=config,
        )

        OmegaConf.save(self.config, self.log_dir / 'config.yaml')

    def get_previous_config(self):
        full_config = OmegaConf.load(self.log_dir.joinpath('config.yaml'))
        return full_config

    def save_checkpoint(self, epoch, model, optimizer, scheduler):
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()},
                   self.checkpoints_dir.joinpath('{:05d}.tar'.format(epoch)))

    def on_epoch_finished(self, epoch):
        self.epoch_start_datetimes.append(datetime.datetime.now())
        epoch_duration = self.epoch_start_datetimes[-1] - self.epoch_start_datetimes[-2]
        avg_duration_s = np.asarray([(self.epoch_start_datetimes[i+1] - self.epoch_start_datetimes[i]).total_seconds()
                                     for i in range(len(self.epoch_start_datetimes) - 1)])
        avg_duration_s = avg_duration_s.mean()
        run_total_epochs = self.config.train.n_epochs
        remaining_datetime = avg_duration_s * (run_total_epochs - epoch - 1)
        remaining_datetime = datetime.timedelta(seconds=int(remaining_datetime))
        
        if self.verbosity >= 1:
            print("End of epoch {} ({}/{}). Duration={:.1f}s, avg={:.1f}s. Estimated remaining time: {} ({})"
                  .format(epoch, epoch + 1, run_total_epochs,
                          epoch_duration.total_seconds(), avg_duration_s,
                          remaining_datetime, humanize.naturaldelta(remaining_datetime)))

    def on_training_finished(self):
        self.tensorboard.flush()
        self.tensorboard.close()
        if self.verbosity >= 1:
            print("[RunLogger] Training has finished")
