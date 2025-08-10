import torch

from data import dataset
from data.sampler import build_subset_samplers


def get_dataset(dataset_cfg):
    if dataset_cfg.name == 'dexed':
        full_dataset = dataset.DexedDataset(**dataset_cfg)
    elif dataset_cfg.name == 'surge':
        full_dataset = dataset.SurgeDataset(**dataset_cfg)
        
    return full_dataset


def get_split_dataloaders(train_cfg, full_dataset):
    subset_samplers = build_subset_samplers(
        full_dataset,
        k_fold=train_cfg.current_k_fold,
        k_folds_count=train_cfg.k_folds,
        test_holdout_proportion=train_cfg.test_holdout_proportion
    )
    dataloaders = dict()
    sub_datasets_lengths = dict()

    for k, sampler in subset_samplers.items():
        drop_last = True
        dataloaders[k] = torch.utils.data.DataLoader(
            full_dataset,
            batch_size=train_cfg.minibatch_size,
            sampler=sampler,
            drop_last=drop_last,
            num_workers=1,
            pin_memory=True,
        )

        sub_datasets_lengths[k] = len(sampler.indices)

        print(
            f"[data/build.py] Dataset '{k}' contains",
            f"{sub_datasets_lengths[k]}/{len(full_dataset)} samples",
            f"({100.0 * sub_datasets_lengths[k]/len(full_dataset):.1f}%)",
        )
            
    return dataloaders

