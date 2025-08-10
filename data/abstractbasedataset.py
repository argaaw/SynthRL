import os
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Sequence

from datetime import datetime
from torchaudio import transforms as T
from tqdm import tqdm

import torch
import torch.utils
import numpy as np

from data.preset import PresetsParams, PresetIndexesHelper


class PresetDataset(torch.utils.data.Dataset, ABC):
    def __init__(
        self,
        note_duration,
        n_fft,
        fft_hop,
        n_mel_bins=128,
        midi_notes=((60, 85),),
        normalize_audio=False,
        dataset_dir=None,
        sample_rate=22050,
    ):
        """
        Abstract Base Class for any synthesizer presets dataset.
        :param note_duration: Tuple: MIDI Note (on_duration, off_duration) in seconds
        :param n_fft: Width of the FFT window for spectrogram computation
        :param fft_hop: STFT hop length (in samples)
        :param midi_notes: Tuple of (midi_pitch, midi_velocity) tuples of notes that should be rendered. Length
            of this tuple is the number of spectrograms that will be fed to the encoder.
        :param n_mel_bins: Number of frequency bins for the Mel-spectrogram. If -1, the normal STFT will be used
        :param normalize_audio:  If True, audio from RenderMan will be normalized
        """
        self.note_duration = note_duration
        self.n_fft = n_fft
        self.fft_hop = fft_hop
        self.n_mel_bins = n_mel_bins
        self.midi_notes = midi_notes
        self.normalize_audio = normalize_audio
        self.valid_preset_UIDs = np.zeros((0,))
        self.learnable_params_idx = list()
        self.sample_rate = sample_rate
        self.dataset_dir = Path(dataset_dir)
        self.wav_files_dir = self.dataset_dir / 'wav'
        self.spec_files_dir = self.dataset_dir / 'spectrogram'
        self.params_path = self.dataset_dir / 'params.pt'
        self.spectrogram = T.MelSpectrogram(sample_rate, n_fft, n_fft, fft_hop, n_mels=n_mel_bins, norm='slaney')

    @property
    @abstractmethod
    def synth_name(self):
        pass

    def __str__(self):
        return "Dataset of {}/{} {} presets. Total items count {}: {} MIDI notes / preset\.\n" \
               "{} learnable synth params, {} fixed params.\n" \
               "Melpectrogram items, size={}." \
            .format(self.valid_presets_count, self.total_nb_presets, self.synth_name,
                    len(self), self.midi_notes_per_preset,
                    len(self.learnable_params_idx), self.total_nb_params - len(self.learnable_params_idx),
                    self.get_spectrogram_tensor_size())

    def __len__(self):
        return self.valid_presets_count * self.midi_notes_per_preset

    def __getitem__(self, i):
        """ Returns a tuple containing a 2D scaled dB spectrograms tensor
        (1st dim: MIDI note, 2nd dim: freq; 2rd dim: time),
        a 1D tensor of parameter values in [0;1],
        and a 1d tensor with remaining int info (preset UID, midi note, vel).

        If this dataset generates audio directly from the synth, only 1 dataloader is allowed.
        A 30000 presets dataset require approx. 7 minutes to be generated on 1 CPU. """
        if self.midi_notes_per_preset > 1:
            preset_index = i // self.midi_notes_per_preset
            midi_note_indexes = [i % self.midi_notes_per_preset]
        else:
            preset_index = i
            midi_note_indexes = range(self.midi_notes_per_preset)
        
        if len(midi_note_indexes) == 1:
            ref_midi_pitch, ref_midi_velocity = self.midi_notes[midi_note_indexes[0]]
        else:
            ref_midi_pitch, ref_midi_velocity = self.midi_notes[0]

        preset_UID = self.valid_preset_UIDs[preset_index]
        waveform, spectrogram = self.get_data_from_file(preset_UID, ref_midi_pitch, ref_midi_velocity)
        synth_param = self.preset_params[preset_index]
        
        return waveform, spectrogram, synth_param, preset_UID, preset_index

    @property
    @abstractmethod
    def total_nb_presets(self):
        """ Total number of presets in the original database, which might be greater than the number of
        available presets in this dataset (some presets can be excluded from learning). """
        pass

    @property
    def valid_presets_count(self):
        """ Total number of presets currently available from this dataset. """
        return len(self.valid_preset_UIDs)

    def get_index_from_preset_UID(self, preset_UID):
        """ Returns the dataset index (or list of indexes) of a preset described by its UID. """
        try:
            index_in_valid_list = list(self.valid_preset_UIDs).index(preset_UID)
        except ValueError:
            raise ValueError("Preset UID {} is not a valid preset UID (it might have been excluded from this dataset)"
                             .format(preset_UID))
        # Check: are there multiple MIDI notes per preset? (dataset size artificial increase)
        if self.midi_notes_per_preset > 1:
            base_index = index_in_valid_list * self.midi_notes_per_preset
            return [base_index + i for i in range(self.midi_notes_per_preset)]
        else:  # 'usual' case: each UID has its own unique dataset index
            return index_in_valid_list

    @property
    def default_midi_note(self):
        """ Default MIDI pitch and velocity, e.g. for audio renders evaluation, labelling, ... """
        return 60, 85

    @property
    def midi_notes_per_preset(self):
        """ Number of available midi notes (different pitch and/or velocity) for a given preset. """
        return len(self.midi_notes)

    @abstractmethod
    def get_full_preset_params(self, preset_UID) -> PresetsParams:
        """ Returns a PresetsParams instance (see preset.py) of 1 preset for the requested preset_UID """
        pass

    @property
    def preset_param_names(self):
        """ Returns a List which contains the name of all parameters of presets (free and constrained). """
        return ['unnamed_param_{}'.format(i) for i in range(self.total_nb_params)]

    def get_preset_param_cardinality(self, idx, learnable_representation=True):
        """ Returns the cardinality i.e. the number of possible different values of all parameters.
        A -1 cardinal indicates a continuous parameter.
        :param idx: The full-preset (VSTi representation) index
        :param learnable_representation: Some parameters can have a reduced cardinality for learning
        (and their learnable representation is scaled consequently). """
        return -1  # Default: continuous params only

    def get_preset_param_quantized_steps(self, idx, learnable_representation=True):
        """ Returns a numpy array of possible quantized values of a discrete parameter. Quantized values correspond
        to floating-point VSTi control values. Returns None if idx refers to a continuous parameter. """
        card = self.get_preset_param_cardinality(idx, learnable_representation)
        if card == -1:
            return None
        elif card == 1:  # Constrained one-value parameter
            return np.asarray([0.5])
        elif card >= 2:
            return np.linspace(0.0, 1.0, endpoint=True, num=card)
        else:
            raise ValueError("Invalid parameter cardinality {}".format(card))

    @property
    def learnable_params_count(self):
        """ Number of learnable VSTi controls. """
        return len(self.learnable_params_idx)

    @property
    def learnable_params_tensor_length(self):
        """ Length of a learnable parameters tensor (contains single-element numerical values and one-hot encoded
        categorical params). """
        _, _, params, _, _ = self.__getitem__(0)
        return params.shape[0]

    @property
    def vst_param_learnable_model(self):
        """ List of models for full-preset (VSTi-compatible) parameters. Possible values are None for non-learnable
        parameters, 'num' for numerical data (continuous or discrete) and 'cat' for categorical data. """
        return ['num' for _ in range(self.total_nb_params)]  # Default: 'num' only

    @property
    def numerical_vst_params(self):
        """ List of indexes of numerical parameters (whatever their discrete number of values) in the VSTi.
        E.g. a 8-step volume param is numerical, while a LFO shape param is not (it is categorical). The
        learnable model can be different from the VSTi model. """
        return [i for i in range(self.total_nb_params)]  # Default: numerical only

    @property
    def categorical_vst_params(self):
        """ List of indexes of categorical parameters in the VSTi. The learnable model can be different
        from the VSTi model."""
        return []  # Default: no categorical params

    @property
    def params_default_values(self):
        """ Dict of default values of VSTi parameters. Not all indexes are keys of this dict (many params do not
        have a default value). """
        return {}

    @property
    @abstractmethod
    def total_nb_params(self):
        """ Total count of constrained and free VST parameters of a preset. """
        pass

    @property
    def preset_indexes_helper(self):
        """ Returns the data.preset.PresetIndexesHelper instance which helps convert full/learnable presets
        from this dataset. """
        return PresetIndexesHelper(nb_params=self.total_nb_params)  # Default: identity

    @abstractmethod
    def _render_audio(self, preset_params: Sequence, midi_note: int, midi_velocity: int):
        """ Renders audio on-the-fly and returns the computed audio waveform and sampling rate.

        :param preset_params: List of preset VST parameters, constrained (constraints from this class ctor
            args must have been applied before passing preset_params).
        """
        pass

    @abstractmethod
    def get_wav_file(self, preset_UID, midi_note, midi_velocity):
        pass

    @abstractmethod
    def get_spec_file(self, preset_UID, midi_note, midi_velocity):
        pass
    
    @abstractmethod
    def get_spec_file_path(self, preset_UID, midi_note, midi_velocity):
        pass
    
    @abstractmethod
    def get_data_from_file(self, preset_UID, midi_pitch, midi_velocity):
        pass

    def _get_wav_file(self, preset_UID):
        """ Returns the preset_UID audio (numpy array). MIDI note and velocity of the note are the class defaults. """
        # FIXME incompatible with future multi-MIDI notes input
        return self.get_wav_file(preset_UID, self.midi_note, self.midi_velocity)

    def get_spectrogram_tensor_size(self):
        """ Returns the size of the first tensor (2D image) returned by this dataset. """
        dummy_spectrogram, _, _, _ = self.__getitem__(0)
        return dummy_spectrogram.size()
    
    def generate_melspec_files(self):
        t_start = datetime.now()
        os.makedirs(self.spec_files_dir, exist_ok=True)
        # MKL and/or PyTorch do not use hyper-threading, and it gives better results... don't use multi-proc here
        workers_args = self._get_multi_note_workers_args(num_workers=1)
        self._generate_melspec_files_batch(workers_args[0])
        delta_t = (datetime.now() - t_start).total_seconds()
        print("Results from {} spectrograms written to {} .pt ({:.1f} minutes total)"
              .format(len(self.valid_preset_UIDs), delta_t/60.0))

    def _generate_melspec_files_batch(self, worker_args):
        for i, args in tqdm(enumerate(worker_args), total=len(worker_args)):
            preset_UID, midi_pitch, midi_velocity = args[0]
            x_wav = self.get_wav_file(preset_UID, midi_pitch, midi_velocity)
            tensor_spectrogram = self.spectrogram(torch.FloatTensor(x_wav))
            tensor_spectrogram = torch.clip(torch.log(tensor_spectrogram + 1e-5) / 12., -1, 1)
            torch.save(tensor_spectrogram, self.get_spec_file_path(preset_UID, midi_pitch, midi_velocity))

    def _get_multi_note_workers_args(self, num_workers):
        """
        Divide all notes to be rendered for all presets into lists (1 list of tuples per worker)
        each tuple contains (preset_UID, midi_pitch, midi_vel). We split the all presets UID across
        workers, and all midi notes for that UID are rendered (or analyzed, ...) by the assigned worker

        :returns: a list of lists of (preset_UID, midi_pitch, midi_vel) tuples to provide to a pool of workers
        """
        # This split returns empty arrays if more workers than UIDs - else, evenly split sub-arrays
        split_preset_UIDs = np.array_split(self.valid_preset_UIDs, num_workers)
        workers_args = list()
        for worker_idx, preset_UIDs in enumerate(split_preset_UIDs):
            workers_args.append(list())  # New worker argument
            for preset_UID in preset_UIDs:
                for midi_pitch, midi_vel in self.midi_notes:
                    workers_args[worker_idx].append([(preset_UID, midi_pitch, midi_vel), worker_idx])
        return workers_args
