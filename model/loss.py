import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, Sequence, Optional

from utils.probability import GaussianKernelConv
from data.preset import PresetIndexesHelper
from data.dexeddataset import DexedDataset


class ParameterLoss:
    """ A 'dynamic' loss which handles different representations of learnable synth parameters
    (numerical and categorical). The appropriate loss can be computed by passing a PresetIndexesHelper instance
    to this class constructor.

    The categorical loss is categorical cross-entropy. """
    def __init__(
        self,
        idx_helper: PresetIndexesHelper,
        categorical_loss_factor: float = 0.2,
        cat_softmax_t: float = 0.1,
        label_smoothing: bool = True,
    ):
        """

        :param idx_helper: PresetIndexesHelper instance, created by a PresetDatabase, to convert vst<->learnable params
        :param categorical_loss_factor: Factor to be applied to the categorical cross-entropy loss, which is
            much greater than the 'corresponding' MSE loss (if the parameter was learned as numerical)
        :param cat_softmax_t: Temperature of the softmax activation applied to cat parameters
        """
        self.idx_helper = idx_helper
        self.cat_softmax_t = cat_softmax_t
        self.cat_loss_factor = categorical_loss_factor
        self.label_smoothing = label_smoothing
        
        if label_smoothing:
            self.gaussian_kernel_conv = GaussianKernelConv()

        # Numerical loss criterion
        self.numerical_criterion = nn.MSELoss(reduction='mean')

        # Pre-compute indexes lists (to use less CPU). 'num' stands for 'numerical' (not number)
        self.num_indexes = self.idx_helper.get_numerical_learnable_indexes()
        self.cat_indexes = self.idx_helper.get_categorical_learnable_indexes()
        self.num_as_cat_indexes = self.idx_helper.num_idx_learned_as_cat.values()

    def __call__(self, u_out: torch.Tensor, u_in: torch.Tensor):
        """ Categorical parameters must be one-hot encoded. """
        # At first: we search for useless parameters (whose loss should not be back-propagated)
        useless_num_learn_param_indexes, useless_cat_learn_param_indexes = list(), list()
        batch_size = u_in.shape[0]
        
        for row in range(batch_size):
            num_indexes, cat_indexes = self.idx_helper.get_useless_learned_params_indexes(u_in[row, :])
            useless_num_learn_param_indexes.append(num_indexes)
            useless_cat_learn_param_indexes.append(cat_indexes)

        # Numerical loss
        num_loss = 0.0

        if len(self.num_indexes) > 0:
            # apply a 0.0 factor for disabled parameters (e.g. Dexed operator w/ output level 0.0)
            for row in range(u_in.shape[0]):
                for num_idx in self.num_indexes:
                    if num_idx in useless_num_learn_param_indexes[row]:
                        u_in[row, num_idx] = 0.0
                        u_out[row, num_idx] = 0.0
            num_loss = self.numerical_criterion(u_out[:, self.num_indexes], u_in[:, self.num_indexes])

        # Categorical loss
        cat_loss = 0.0

        if len(self.cat_indexes) > 0:
            # For each categorical output (separate loss computations...)
            for cat_learn_indexes in self.cat_indexes:
                # don't compute cat loss for disabled parameters (e.g. Dexed operator w/ output level 0.0)
                rows_to_remove = list()
                for row in range(batch_size):  # Need to check cat index 0 only
                    if cat_learn_indexes[0] in useless_cat_learn_param_indexes[row]:
                        rows_to_remove.append(row)
                useful_rows = None  # None means that the full batch is useful
                if len(rows_to_remove) > 0:  # If this batch contains useless inferred params
                    useful_rows = list(range(0, batch_size))
                    for row in rows_to_remove:
                        useful_rows.remove(row)
                # Direct cross-entropy computation. The one-hot target is used to select only q output probabilities
                # corresponding to target classes with p=1. We only need a limited number of output probabilities
                # (they actually all depend on each other thanks to the softmax output layer).
                target_one_hot = u_in[:, cat_learn_indexes]
                
                if self.label_smoothing and cat_learn_indexes in self.num_as_cat_indexes:
                    target_one_hot = self.gaussian_kernel_conv(target_one_hot)

                if useful_rows is not None:  # Some rows can be discarded from loss computation
                    target_one_hot = target_one_hot[useful_rows, :]
                
                q_odds = u_out[:, cat_learn_indexes]
                
                # The same rows must be discarded from loss computation (if the preset didn't use this cat param)
                if useful_rows is not None:
                    q_odds = q_odds[useful_rows, :]
                
                # softmax T° if required: q_odds might not sum to 1.0 already if no softmax was applied before
                q_odds = torch.softmax(q_odds / self.cat_softmax_t, dim=1)
                # Then the cross-entropy can be computed
                # batch-sum and normalization vs. batch size
                param_cat_loss = - torch.sum(target_one_hot * torch.log(q_odds)) / (batch_size - len(rows_to_remove))

                # Add the temp per-param loss
                cat_loss += param_cat_loss

            cat_loss = cat_loss / len(self.cat_indexes)

        # Losses weighting - Cross-Entropy is usually be much bigger than MSE. num_loss
        return num_loss + cat_loss * self.cat_loss_factor


class QuantizedNumericalParamsLoss:
    """ 'Quantized' parameters loss: to get a meaningful (but non-differentiable) loss, inferred parameter
    values must be quantized as they would be in the synthesizer.

    Only numerical parameters are involved in this loss computation. The PresetIndexesHelper ctor argument
    allows this class to know which params are numerical.
    The loss to be applied after quantization can be passed as a ctor argument.

    This loss breaks the computation path (.backward cannot be applied to it).
    """
    def __init__(self, idx_helper: PresetIndexesHelper, numerical_loss=nn.MSELoss(),
                 reduce: bool = True, limited_vst_params_indexes: Optional[Sequence] = None):
        """

        :param idx_helper:
        :param numerical_loss:
        :param limited_vst_params_indexes: List of VST params to include into to the loss computation. Can be uses
            to measure performance of specific groups of params. Set to None to include all numerical parameters.
        """
        self.idx_helper = idx_helper
        self.numerical_loss = numerical_loss
        # Cardinality checks
        for vst_idx, _ in self.idx_helper.num_idx_learned_as_cat.items():
            assert self.idx_helper.vst_param_cardinals[vst_idx] > 0
        # Number of numerical parameters considered for this loss (after cat->num conversions). For tensor pre-alloc
        self.num_params_count = len(self.idx_helper.num_idx_learned_as_num)\
                                + len(self.idx_helper.num_idx_learned_as_cat)
        self.limited_vst_params_indexes = limited_vst_params_indexes
        self.reduce = reduce

    def __call__(self, u_out: torch.Tensor, u_in: torch.Tensor):
        """ Returns the loss for numerical VST params only (searched in u_in and u_out).
        Learnable representations can be numerical (in [0.0, 1.0]) or one-hot categorical.
        The type of representation has been stored in self.idx_helper """
        errors = dict()
        # Partial tensors (for final loss computation)
        minibatch_size = u_in.size(0)
        # pre-allocate tensors
        u_in_num = torch.empty((minibatch_size, self.num_params_count), device=u_in.device, requires_grad=False)
        u_out_num = torch.empty((minibatch_size, self.num_params_count), device=u_in.device, requires_grad=False)
        # if limited vst indexes: fill with zeros (some allocated cols won't be used). Slow but used for eval only.
        if self.limited_vst_params_indexes is not None:
            u_in_num[:, :], u_out_num[:, :] = 0.0, 0.0
        # Column-by-column tensors filling
        cur_num_tensors_col = 0
        # quantize numerical learnable representations
        for vst_idx, learn_idx in self.idx_helper.num_idx_learned_as_num.items():
            if self.limited_vst_params_indexes is not None:  # if limited vst indexes:
                if vst_idx not in self.limited_vst_params_indexes:  # continue if this param is not included
                    continue
            param_batch = u_in[:, learn_idx].detach()
            u_in_num[:, cur_num_tensors_col] = param_batch  # Data copy - does not modify u_in
            param_batch = u_out[:, learn_idx].detach().clone()
            if self.idx_helper.vst_param_cardinals[vst_idx] > 0:  # don't quantize <0 cardinal (continuous)
                cardinal = self.idx_helper.vst_param_cardinals[vst_idx]
                param_batch = torch.round(param_batch * (cardinal - 1.0)) / (cardinal - 1.0)
            u_out_num[:, cur_num_tensors_col] = param_batch
            errors[vst_idx] = self.numerical_loss(u_in_num[:, cur_num_tensors_col], u_out_num[:, cur_num_tensors_col]).item()
            cur_num_tensors_col += 1
        # convert one-hot encoded learnable representations of (quantized) numerical VST params
        for vst_idx, learn_indexes in self.idx_helper.num_idx_learned_as_cat.items():
            if self.limited_vst_params_indexes is not None:  # if limited vst indexes:
                if vst_idx not in self.limited_vst_params_indexes:  # continue if this param is not included
                    continue
            cardinal = len(learn_indexes)
            # Classes as column-vectors (for concatenation)
            in_classes = torch.argmax(u_in[:, learn_indexes], dim=-1).detach().type(torch.float)
            u_in_num[:, cur_num_tensors_col] = in_classes / (cardinal-1.0)
            out_classes = torch.argmax(u_out[:, learn_indexes], dim=-1).detach().type(torch.float)
            u_out_num[:, cur_num_tensors_col] = out_classes / (cardinal-1.0)
            errors[vst_idx] = self.numerical_loss(u_in_num[:, cur_num_tensors_col], u_out_num[:, cur_num_tensors_col]).item()
            cur_num_tensors_col += 1
        # Final size checks
        if self.limited_vst_params_indexes is None:
            assert cur_num_tensors_col == self.num_params_count
        else:
            pass  # No size check for limited params (a list with unlearned and/or cat params can be provided)
            #  assert cur_num_tensors_col == len(self.limited_vst_params_indexes)
        
        if self.reduce:
            return self.numerical_loss(u_out_num, u_in_num)  # Positive diff. if output > input
        else:
            return errors, self.numerical_loss(u_out_num, u_in_num)


class CategoricalParamsAccuracy:
    """ Only categorical parameters are involved in this loss computation. """
    def __init__(self, idx_helper: PresetIndexesHelper, reduce=True, percentage_output=True,
                 limited_vst_params_indexes: Optional[Sequence] = None):
        """
        :param idx_helper: allows this class to know which params are categorical
        :param reduce: If True, an averaged accuracy will be returned. If False, a dict of accuracies (keys =
          vst param indexes) is returned.
        :param percentage_output: If True, accuracies in [0.0, 100.0], else in [0.0, 1.0]
        :param limited_vst_params_indexes: List of VST params to include into to the loss computation. Can be uses
            to measure performance of specific groups of params. Set to None to include all numerical parameters.
        """
        self.idx_helper = idx_helper
        self.reduce = reduce
        self.percentage_output = percentage_output
        self.limited_vst_params_indexes = limited_vst_params_indexes

    def __call__(self, u_out: torch.Tensor, u_in: torch.Tensor):
        """ Returns accuracy (or accuracies) for all categorical VST params.
        Learnable representations can be numerical (in [0.0, 1.0]) or one-hot categorical.
        The type of representation is stored in self.idx_helper """
        accuracies = dict()
        # Accuracy of numerical learnable representations (involves quantization)
        for vst_idx, learn_idx in self.idx_helper.cat_idx_learned_as_num.items():
            if self.limited_vst_params_indexes is not None:  # if limited vst indexes:
                if vst_idx not in self.limited_vst_params_indexes:  # continue if this param is not included
                    continue
            cardinal = self.idx_helper.vst_param_cardinals[vst_idx]
            param_batch = torch.unsqueeze(u_in[:, learn_idx].detach(), 1)  # Column-vector
            # Class indexes, from 0 to cardinal-1
            target_classes = torch.round(param_batch * (cardinal - 1.0)).type(torch.int32)  # New tensor allocated
            param_batch = torch.unsqueeze(u_out[:, learn_idx].detach(), 1)
            out_classes = torch.round(param_batch * (cardinal - 1.0)).type(torch.int32)  # New tensor allocated
            accuracies[vst_idx] = (target_classes == out_classes).count_nonzero().item() / target_classes.numel()
        # accuracy of one-hot encoded categorical learnable representations
        for vst_idx, learn_indexes in self.idx_helper.cat_idx_learned_as_cat.items():
            if self.limited_vst_params_indexes is not None:  # if limited vst indexes:
                if vst_idx not in self.limited_vst_params_indexes:  # continue if this param is not included
                    continue
            target_classes = torch.argmax(u_in[:, learn_indexes], dim=-1)  # New tensor allocated
            out_classes = torch.argmax(u_out[:, learn_indexes], dim=-1)  # New tensor allocated
            accuracies[vst_idx] = (target_classes == out_classes).count_nonzero().item() / target_classes.numel()
        # Factor 100.0?
        if self.percentage_output:
            for k, v in accuracies.items():
                accuracies[k] = v * 100.0
        # Reduction if required
        if self.reduce:
            return np.asarray([v for _, v in accuracies.items()]).mean()
        else:
            return accuracies


class PresetProcessor:
    """ A 'dynamic' loss which handles different representations of learnable synth parameters
    (numerical and categorical). The appropriate loss can be computed by passing a PresetIndexesHelper instance
    to this class constructor.

    The categorical loss is categorical cross-entropy. """
    def __init__(
        self,
        dataset: DexedDataset,
        idx_helper: PresetIndexesHelper,
        device: str,
        cat_softmax_t: float = 0.1,
    ):
        """
        :param idx_helper: PresetIndexesHelper instance, created by a PresetDatabase, to convert vst<->learnable params
        :param cat_softmax_t: Temperature of the softmax activation applied to cat parameters
        """
        self.params_default_values = dataset.params_default_values
        self.idx_helper = idx_helper
        self.device = device
        self.cat_softmax_t = cat_softmax_t
        self.cat_indexes = self.idx_helper.get_categorical_learnable_indexes()

    def __call__(self, u_out: torch.Tensor, deterministic: bool = False):
        """ Categorical parameters must be one-hot encoded. """
        batch_size = u_out.shape[0]
        n_params = self.idx_helper.full_preset_size
        full_presets = -0.1 * torch.ones((batch_size, n_params))
        full_actions = torch.zeros((batch_size, n_params), dtype=torch.int64, device=u_out.device)

        for vst_idx, learnable_indexes in enumerate(self.idx_helper.full_to_learnable):
            if self.idx_helper.vst_param_learnable_model[vst_idx] is None:
                full_presets[:, vst_idx] = self.params_default_values[vst_idx] * torch.ones((batch_size, ))
            elif isinstance(learnable_indexes, Iterable):
                with torch.no_grad():
                    logits = u_out[:, learnable_indexes]
                    probs = torch.softmax(logits / self.cat_softmax_t, dim=1)
                    if deterministic:
                        actions = torch.argmax(probs, dim=-1)
                    else:
                        actions = torch.multinomial(probs, num_samples=1).squeeze()
                n_classes = self.idx_helper.vst_param_cardinals[vst_idx]
                full_actions[:, vst_idx] = actions
                full_presets[:, vst_idx] = actions / (n_classes - 1.0)
            elif isinstance(learnable_indexes, int):
                full_presets[:, vst_idx] = u_out[:, learnable_indexes]
            else:
                raise ValueError("Bad learnable indices for vst idx = {}".format(vst_idx))

        return full_presets, full_actions
    
    def get_mean_log_probs(self, v_out, actions, importance_sampling=False):
        batch_size = actions.shape[0]
        mean_log_probs = torch.zeros((batch_size, 1), device=self.device)

        for vst_idx, learnable_indexes in enumerate(self.idx_helper.full_to_learnable):
            if isinstance(learnable_indexes, Iterable):
                logits = v_out[:, learnable_indexes]
                probs = torch.softmax(logits / self.cat_softmax_t, dim=1)
                action_probs = torch.gather(probs, 1, actions[:, vst_idx].unsqueeze(1))
                log_probs = torch.log(action_probs)
                if importance_sampling:
                    mean_log_probs += action_probs.detach() * log_probs
                else:
                    mean_log_probs += log_probs

        mean_log_probs = mean_log_probs / len(self.cat_indexes)
        return mean_log_probs


def calculate_rewards(
        sc: torch.Tensor,
        log_mae: torch.Tensor,
        mfcc_mae: torch.Tensor,
        sc_coef: float = 0.7,
        mfcc_coef: float = 0.03,
    ):
    
    rewards = sc_coef * sc + (1 - sc_coef - mfcc_coef) * log_mae + mfcc_coef * mfcc_mae
    rewards = (1 / torch.clamp(rewards, min=0.1, max=5.0))
    
    return rewards.unsqueeze(1)
