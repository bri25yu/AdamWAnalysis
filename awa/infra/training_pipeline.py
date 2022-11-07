from typing import Any, Callable, Dict

from abc import ABC, abstractmethod

import os

import time

from tqdm.notebook import trange, tqdm

from tensorboardX import SummaryWriter

from numpy.random import seed as np_seed

from torch import Tensor, from_numpy, no_grad
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
from torch.random import manual_seed as torch_seed

from awa import TRAIN_OUTPUT_DIR, RESULTS_DIR
from awa.infra.env import Env
from awa.infra.modeling import KMeansConfig, KMeansModel


__all__ = ["TrainingPipeline"]


TORCH_DEVICE = "cuda"


class TrainingPipeline(ABC):
    NUM_STEPS = 10000
    BATCH_SIZE = 32
    EVAL_EXAMPLES = 1000
    TEST_EXAMPLES = 10000
    N_CLASSES = 2
    DIM = 2

    @abstractmethod
    def get_optimizer(self, params) -> Optimizer:
        pass

    def run(self, seed: int=42, leave_tqdm=True) -> None:
        num_steps = self.NUM_STEPS
        batch_size = self.BATCH_SIZE

        np_seed(seed)
        torch_seed(seed)

        model = self.get_model()
        optimizer = self.get_optimizer(model.parameters())
        loss_fn = CrossEntropyLoss()
        self.setup_logging(seed)

        model = model.to(TORCH_DEVICE)
        train_data, train_labels, val_data, val_labels, test_data, test_labels = self._get_data()

        for i in trange(num_steps, desc="Training", leave=leave_tqdm):
            batch_data = train_data[batch_size * i: batch_size * (i+1)].to(TORCH_DEVICE)
            batch_labels = train_labels[batch_size * i: batch_size * (i+1)].to(TORCH_DEVICE)

            model.train()
            loss: Tensor = loss_fn(model(batch_data), batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.log({"loss": loss}, "train", i)
            self.log(self.compute_metrics(model, val_data, val_labels, loss_fn), "eval", i)

        self.log(self.compute_metrics(model, test_data, test_labels, loss_fn), "test", i+1)

    def benchmark(self) -> None:
        self.use_benchmark_logging = True

        seeds = [41, 42, 43]
        for seed in tqdm(seeds, desc="Benchmarking"):
            self.run(seed=seed, leave_tqdm=False)

    def _get_data(self):
        """
        Convenience function to get data
        """
        n_classes = self.N_CLASSES
        dim = self.DIM
        num_steps = self.NUM_STEPS
        batch_size = self.BATCH_SIZE
        train_examples = num_steps * batch_size
        eval_examples = self.EVAL_EXAMPLES
        test_examples = self.TEST_EXAMPLES
        total_examples = train_examples + eval_examples + test_examples

        env = Env(total_examples, n_classes, dim)

        train_data = from_numpy(env.points[:train_examples])
        train_labels = from_numpy(env.labels[:train_examples])
        val_data = from_numpy(env.points[train_examples: train_examples + eval_examples])
        val_labels = from_numpy(env.labels[train_examples: train_examples + eval_examples])
        test_data = from_numpy(env.points[train_examples + eval_examples: train_examples + eval_examples + test_examples])
        test_labels = from_numpy(env.labels[train_examples + eval_examples: train_examples + eval_examples + test_examples])

        assert train_data.size()[0] == train_examples
        assert val_data.size()[0] == eval_examples
        assert test_data.size()[0] == test_examples

        val_data = val_data.to(TORCH_DEVICE)
        val_labels = val_labels.to(TORCH_DEVICE)
        test_data = test_data.to(TORCH_DEVICE)
        test_labels = test_labels.to(TORCH_DEVICE)

        return train_data, train_labels, val_data, val_labels, test_data, test_labels

    def get_model(self) -> Module:
        config = KMeansConfig(
            input_dim=self.DIM,
            num_classes=self.N_CLASSES,
            hidden_dim=64,
            n_heads=1024,
            head_dim=64,
        )
        return KMeansModel(config)

    def compute_metrics(self, model: Module, data: Tensor, labels: Tensor, loss_fn: Callable) -> None:
        model.eval()
        with no_grad():
            outputs: Tensor = model(data)
            loss = loss_fn(outputs, labels)
            accuracy = (outputs.argmax(dim=1) == labels).sum() / data.size()[0]

        return {
            "loss": loss,
            "accuracy": accuracy,
        }

    def log(self, logs: Dict[str, Any], prefix: str, step: int) -> None:
        for value_name, value in logs.items():
            self.logger.add_scalar(f"{value_name}_{prefix}", value, step)

        self.logger.flush()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def setup_logging(self, seed: int) -> None:
        if getattr(self, "use_benchmark_logging", False):
            log_dir = os.path.join(RESULTS_DIR, self.name, f"seed={seed}")
        else:
            log_dir = os.path.join(TRAIN_OUTPUT_DIR, self.name, f"seed={seed}", f"run{time.time()}")

        self.logger = SummaryWriter(log_dir)
