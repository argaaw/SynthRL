"""
Samplers for any abstract PresetDataset class, which can be used as train/valid/test samplers.
Support k-fold cross validation and subtleties of multi-note (multi-layer spectrogram) preset datasets.

"""
from typing import Dict

import numpy as np
import torch
import torch.utils.data


def build_subset_samplers(
        dataset: torch.utils.data.Dataset,
        k_fold: int = 0,
        k_folds_count: int = 5,
        test_holdout_proportion: float = 0.2,
        random_seed: int = 0,
    ) -> Dict[str, torch.utils.data.SubsetRandomSampler]:
    """
    Builds 'train', 'validation' and 'test' subset samplers

    :param dataset: Required to properly separate dataset items indexes by preset UIDs (not to split
        a multi-note preset into multiple subsets).
    :param k_fold: Current k-fold cross-validation fold index
    :param k_folds_count: Total number of k-folds
    :param test_holdout_proportion: Proportion of 'test' data, excluded from cross-validation folds.
    :param random_seed: For reproducibility, always use the same seed

    :returns: dict of subset_samplers
    """
    presets_count = dataset.valid_presets_count
    all_preset_indexes = np.arange(presets_count)
    preset_indexes = dict()
    rng = np.random.default_rng(seed=random_seed)

    # Shuffle preset indexes, and separate them into subsets
    rng.shuffle(all_preset_indexes)  # in-place shuffling
    first_test_idx = int(np.floor(presets_count * (1.0 - test_holdout_proportion)))
    non_test_preset_indexes, preset_indexes['test'] = np.split(all_preset_indexes, [first_test_idx])

    # All folds are retrieved - we'll choose only one of these as validation subset, and merge the others
    preset_indexes_folds = np.array_split(non_test_preset_indexes, k_folds_count)
    preset_indexes['validation'] = preset_indexes_folds[k_fold]
    preset_indexes['train'] = np.hstack([preset_indexes_folds[i] for i in range(k_folds_count) if i != k_fold])

    # Final indexes
    if dataset.midi_notes_per_preset == 1 or dataset.multichannel_stacked_spectrograms:
        final_indexes = preset_indexes
    else:  # multi-note, single-layer spectrogram dataset: dataset indexes are not preset indexes
        final_indexes = dict()
        # We don't need to shuffle again these groups (SubsetRandomSampler will do it)
        for k in preset_indexes:  # k: train, valid or test
            final_indexes[k] = list()
            for preset_idx in preset_indexes[k]:
                final_indexes[k] += [preset_idx * dataset.midi_notes_per_preset + i
                                     for i in range(dataset.midi_notes_per_preset)]
                
    subset_samplers = dict()

    for k in final_indexes:
        subset_samplers[k] = torch.utils.data.SubsetRandomSampler(final_indexes[k])
        
    return subset_samplers
