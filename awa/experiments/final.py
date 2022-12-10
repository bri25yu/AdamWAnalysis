import os

import time

from tqdm.notebook import tqdm

from torch.nn import Module

from tensorboardX import SummaryWriter

from awa import TRAIN_OUTPUT_DIR, RESULTS_DIR
from awa.infra import TrainingPipeline, Env
from awa.vis_mixins.no_vis import NoVisMixin
from awa.optimizer_mixins import AdamWOptimizerMixin
from awa.modeling.v2 import ModelConfig, FinalModel


__all__ = ["FinalModelExperiment"]


class FinalModelExperiment(AdamWOptimizerMixin, NoVisMixin, TrainingPipeline):
    BATCH_SIZE = 1024
    LR = 2e-3
    WEIGHT_DECAY = 1e-2
    LEARNING_RATES = [2e-3, 3e-3, 5e-3]

    def get_model(self, env: Env) -> Module:
        return FinalModel(env, ModelConfig())

    def benchmark(self) -> None:
        self.use_benchmark_logging = True

        for lr in tqdm(self.LEARNING_RATES, desc="Benchmarking"):
            self.LR = lr
            self.run(leave_tqdm=False)

    def setup_logging(self, seed: int) -> None:
        if not hasattr(self, "use_benchmark_logging"):
            self.use_benchmark_logging = False

        if self.use_benchmark_logging:
            log_dir = os.path.join(RESULTS_DIR, self.name, f"lr={self.LR:.0e}")
            self.setup_visualization_logging()
        else:
            log_dir = os.path.join(TRAIN_OUTPUT_DIR, self.name, f"lr={self.LR:.0e}", f"run{time.time()}")

        self.logger = SummaryWriter(log_dir)
