"""
Audio utils (spectrograms, G&L phase reconstruction, ...)
"""

import os
import sys
from scipy.signal import resample
from contextlib import contextmanager
from typing import List
import psutil

import numpy as np
import multiprocessing

import torch
import torch.fft
import torch.nn as nn
from torch.utils.data import Dataset
from nnAudio.features import STFT
from nnAudio.features.mel import MFCC


class AudioRenderer():
    def __init__(
            self,
            dataset: Dataset,
            write_sr: int,
            num_workers: int,
            device: str,
            midi_pitch: int = 60,
            midi_velocity: int = 85,
        ):
        self.dataset = dataset
        self.write_sr = write_sr
        self.num_workers = num_workers
        self.device = device
        self.midi_pitch = midi_pitch
        self.midi_velocity = midi_velocity

    def single_process_render(self, full_preset_out: torch.Tensor):
        wav = self._render_audio(full_preset_out)
        return torch.FloatTensor(wav).to(self.device)

    def multi_process_render(self, full_preset_out: torch.Tensor):
        full_preset_out_split = np.array_split(full_preset_out, self.num_workers, axis=0)
        workers_data = []

        for i in range(self.num_workers):
            workers_data.append((full_preset_out_split[i], ))

        with multiprocessing.Pool(self.num_workers) as p:
            wavs_split = p.map(self._render_worker, workers_data)

        inferred_wav = np.vstack(wavs_split)
        return torch.FloatTensor(inferred_wav).to(self.device)

    def _render_worker(self, worker_args: List):
        pid = os.getpid()
        cpus = list(range(psutil.cpu_count()))
        os.sched_setaffinity(pid, cpus)
        return self._render_audio(*worker_args)

    def _render_audio(self, full_preset_out: torch.Tensor):
        wavs = []

        for i in range(full_preset_out.shape[0]):
            with suppress_output():
                x_wav_inferred, Fs = self.dataset._render_audio(
                    full_preset_out[i],
                    self.midi_pitch,
                    self.midi_velocity,
                )
            num_samples = int(len(x_wav_inferred) * self.write_sr / Fs)
            resampled_wav = resample(x_wav_inferred, num_samples)
            wavs.append(resampled_wav)

        return np.array(wavs)
    

class Spectrogram_Processor(nn.Module):
    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        sr: int = 22050,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.eps = eps
        self.mfcc13 = MFCC(
            sr=sr,
            n_mfcc=13,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        self.mfcc40 = MFCC(
            sr=sr,
            n_mfcc=40,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        self.stft = STFT(
            n_fft=n_fft,
            hop_length=hop_length,
            sr=sr,
            output_format='Magnitude'
        )

    def calculate_metrics(self, wav_1: torch.Tensor, wav_2: torch.Tensor):
        bs = wav_1.shape[0]
        wavs = torch.cat((wav_1, wav_2), dim=0)

        mfcc13s = self.mfcc13(wavs)
        mfcc40s = self.mfcc40(wavs)
        mfcc13_mae = torch.abs(mfcc13s[:bs] - mfcc13s[bs:]).mean(dim=[1, 2])
        mfcc40_mae = torch.abs(mfcc40s[:bs] - mfcc40s[bs:]).mean(dim=[1, 2])

        specs = self.stft(wavs)
        specs = torch.clamp(specs, min=self.eps)
        fro = torch.sqrt(torch.sum((specs[:bs] - specs[bs:]) ** 2, dim=[1, 2]))
        fro_gt = torch.sqrt(torch.sum(specs[:bs] ** 2, dim=[1, 2]))
        sc = torch.clamp(fro / fro_gt, max=5.0)
        log_specs = torch.log10(specs)
        log_mae = torch.abs(log_specs[:bs] - log_specs[bs:]).mean(dim=[1, 2])
        
        return sc, log_mae, mfcc13_mae, mfcc40_mae
    

@contextmanager
def suppress_output():
    """Suppress stdout and stderr."""
    # Save the original file descriptors
    original_stdout_fd = sys.stdout.fileno()
    original_stderr_fd = sys.stderr.fileno()

    # Duplicate the original file descriptors (stdout and stderr)
    saved_stdout_fd = os.dup(original_stdout_fd)
    saved_stderr_fd = os.dup(original_stderr_fd)

    # Open a new file descriptor that redirects to /dev/null
    devnull_fd = os.open(os.devnull, os.O_RDWR)

    try:
        # Redirect stdout and stderr to /dev/null
        os.dup2(devnull_fd, original_stdout_fd)
        os.dup2(devnull_fd, original_stderr_fd)

        yield
    finally:
        # Restore the original stdout and stderr
        os.dup2(saved_stdout_fd, original_stdout_fd)
        os.dup2(saved_stderr_fd, original_stderr_fd)

        # Close the duplicated file descriptors
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)
        