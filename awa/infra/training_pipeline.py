from typing import Any, Callable, Dict, Tuple, Union

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
from awa.modeling.base import ModelOutput, ModelBase
from awa.infra.utils import TORCH_DEVICE


__all__ = ["TrainingPipeline"]


class TrainingPipeline(ABC):
    NUM_STEPS = 10000
    EVAL_STEPS = 50  # Eval 10000 / 50 = 200 times
    BATCH_SIZE = 32
    EVAL_EXAMPLES = 10000
    TEST_EXAMPLES = 10000
    N_CLASSES = 2
    DIM = 2

    @abstractmethod
    def get_optimizer(self, params) -> Optimizer:
        pass

    @abstractmethod
    def get_model(self, env: Env) -> ModelBase:
        pass

    @abstractmethod
    def visualize(self, data: Tensor, labels: Tensor) -> None:
        pass

    @abstractmethod
    def store_eval_logs_to_visualize(self, output: ModelOutput, loss: Tensor) -> None:
        pass

    @abstractmethod
    def setup_visualization_logging(self) -> None:
        pass

    def run(self, seed: int=42, leave_tqdm=True) -> None:
        num_steps = self.NUM_STEPS
        eval_steps = self.EVAL_STEPS
        batch_size = self.BATCH_SIZE

        np_seed(seed)
        torch_seed(seed)
        self.setup_logging(seed)

        env, train_data, train_labels, val_data, val_labels, test_data, test_labels = self._get_data()

        model = self.get_model(env)
        model = model.to(TORCH_DEVICE)
        optimizer = self.get_optimizer(model.parameters())
        loss_fn = CrossEntropyLoss()

        for i in trange(num_steps, desc="Training", leave=leave_tqdm):
            batch_data = train_data[batch_size * i: batch_size * (i+1)].to(TORCH_DEVICE)
            batch_labels = train_labels[batch_size * i: batch_size * (i+1)].to(TORCH_DEVICE)

            model.train()
            output: ModelOutput = model(batch_data)
            loss: Tensor = loss_fn(output.logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.log({"loss": loss}, "train", i)
            if (i+1) % eval_steps == 0:
                self.log(self.compute_metrics(model, val_data, val_labels, loss_fn, is_eval=True), "eval", i)

        self.log(self.compute_metrics(model, test_data, test_labels, loss_fn), "test", i+1)
        if self.use_benchmark_logging:
            self.visualize(val_data, val_labels)

    def benchmark(self) -> None:
        self.use_benchmark_logging = True

        # Work with one seed for now for visualization purposes
        # seeds = [41, 42, 43]
        seeds = [42]
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

        to_log = [("train", train_labels), ("val", val_labels), ("test", test_labels)]
        for prefix, labels in to_log:
            self.log({
                "mle_accuracy": labels.unique(return_counts=True)[1].max() / labels.size()[0],
            }, prefix)

        assert train_data.size()[0] == train_examples
        assert val_data.size()[0] == eval_examples
        assert test_data.size()[0] == test_examples

        val_data = val_data.to(TORCH_DEVICE)
        val_labels = val_labels.to(TORCH_DEVICE)
        test_data = test_data.to(TORCH_DEVICE)
        test_labels = test_labels.to(TORCH_DEVICE)

        return env, train_data, train_labels, val_data, val_labels, test_data, test_labels

    def compute_metrics(
        self, model: Module, data: Tensor, labels: Tensor, loss_fn: Callable, is_eval: bool=False
    ) -> Union[Dict[str, Tensor], Tuple[Dict[str, Tensor], Tensor]]:
        model.eval()
        with no_grad():
            output: ModelOutput = model(data)
            loss = loss_fn(output.logits, labels)
            predicted_labels = output.logits.argmax(dim=1)
            accuracy = (predicted_labels == labels).sum() / data.size()[0]

        if self.use_benchmark_logging and is_eval:
            self.store_eval_logs_to_visualize(output, loss)

        logs = output.logs if output.logs else {}

        return {
            "loss": loss,
            "accuracy": accuracy,
            **logs,
        }

    def log(self, logs: Dict[str, Any], prefix: str="", step: int=0) -> None:
        for value_name, value in logs.items():
            if value_name.endswith("_weight"):
                self.logger.add_histogram(value_name, value, step)
            elif value.size() in ((), (1,)):
                self.logger.add_scalar(f"{value_name}_{prefix}", value, step)
            else:
                raise NotImplementedError

        self.logger.flush()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def setup_logging(self, seed: int) -> None:
        if not hasattr(self, "use_benchmark_logging"):
            self.use_benchmark_logging = False

        if self.use_benchmark_logging:
            log_dir = os.path.join(RESULTS_DIR, self.name, f"seed={seed}")
            self.setup_visualization_logging()
        else:
            log_dir = os.path.join(TRAIN_OUTPUT_DIR, self.name, f"seed={seed}", f"run{time.time()}")

        self.logger = SummaryWriter(log_dir)
