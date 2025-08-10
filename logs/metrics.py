import numpy as np
import torch


class SimpleMetric:
    """ A very simple class for storing a metric, which provides EpochMetric-compatible methods """
    def __init__(self, value=0.0):
        if isinstance(value, torch.Tensor):
            self._value = value.item()
        else:
            self._value = value

    def on_new_epoch(self):
        return None

    def get(self):
        return self._value

    @property
    def value(self):
        return self.get()


class EpochMetric:
    """ Can store mini-batch metric values in order to compute an epoch-averaged metric. """
    def __init__(self, normalized_losses=True):
        """
        :param normalized_losses: If False, the mini-batch size must be given when data is appended
        """
        # :param epoch_end_metric: If given, this class will append end-of-epoch values to this BufferedMetric instance
        self.normalized_losses = normalized_losses
        self.buffer = list()

    def on_new_epoch(self):
        self.buffer = list()

    def append(self, value, minibatch_size=-1):
        if minibatch_size <= 0:
            assert self.normalized_losses is True
        if isinstance(value, torch.Tensor):
            self.buffer.append(value.item())
        else:
            self.buffer.append(value)

    def get(self):
        """ Returns the mean of values stored since last call to on_new_epoch() """
        if len(self.buffer) == 0:
            raise ValueError()
        return np.asarray(self.buffer).mean()

    @property
    def value(self):
        return self.get()
