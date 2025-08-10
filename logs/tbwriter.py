
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams


class CorrectedSummaryWriter(SummaryWriter):
    """ SummaryWriter corrected to prevent extra runs to be created
    in Tensorboard when adding hparams.

    Original code in torch/utils/tensorboard.writer.py,
    modification by method overloading inspired by https://github.com/pytorch/pytorch/issues/32651 """

    def add_hparams(self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None):
        assert run_name is None  # Disabled feature. Run name init by summary writer ctor

        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

        # run_name argument is discarded and the writer itself is used (no extra writer instantiation)
        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v)


class TensorboardSummaryWriter(CorrectedSummaryWriter):
    """ Tensorboard SummaryWriter with corrected add_hparams method
     and extra functionalities. """

    def __init__(
        self,
        log_dir=None,
        comment='',
        purge_step=None,
        max_queue=10,
        flush_secs=120,
        filename_suffix='',
        config=None,
    ):
        super().__init__(log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)
        # Full-Config is required. Default constructor values allow to keep the same first constructor args
        self.config = config
        
    def init_hparams_and_metrics(self, metrics):
        """ Hparams and Metric initialization. Will pass if training resumes from saved checkpoint.
        Hparams will be definitely set but metrics can be updated during training.

        :param metrics: Dict of BufferedMetric
        """
        self.update_metrics(metrics)

    def update_metrics(self, metrics):
        """ Updates Tensorboard metrics

        :param metrics: Dict of values and/or BufferedMetric instances
        :return: None
        """
        metrics_dict = dict()
        for k, metric in metrics.items():
            metrics_dict[k] = metric
        self.add_hparams(self.hyper_params, metrics_dict, hparam_domain_discrete=None)

