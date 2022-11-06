from abc import ABC, abstractmethod

import os

import time

from itertools import chain

from tqdm.notebook import trange

from tensorboardX import SummaryWriter

from numpy.random import np_seed

from torch import Tensor, from_numpy, no_grad
from torch.nn import Module, CrossEntropyLoss, ReLU, Sequential, Linear, Identity
from torch.optim import Optimizer
from torch.random import seed as torch_seed

from awa import TRAIN_OUTPUT_DIR, RESULTS_DIR
from awa.infra.env import Env


__all__ = ["TrainingPipeline"]


class TrainingPipeline(ABC):
    NUM_STEPS = 1000
    BATCH_SIZE = 32
    EVAL_EXAMPLES = 1000
    TEST_EXAMPLES = 10000

    @abstractmethod
    def get_optimizer(self, params) -> Optimizer:
        pass

    def run(self, seed: int=42, leave_tqdm=True, base_log_dir=TRAIN_OUTPUT_DIR) -> None:
        num_steps = self.NUM_STEPS
        batch_size = self.BATCH_SIZE
        train_examples = num_steps * batch_size
        eval_examples = self.EVAL_EXAMPLES
        test_examples = self.TEST_EXAMPLES
        total_examples = train_examples + eval_examples + test_examples

        np_seed(seed)
        torch_seed(seed)

        env = Env(total_examples, 2)
        model = self.get_model()
        optimizer = self.get_optimizer()
        loss_fn = CrossEntropyLoss()
        self.setup_logging(base_log_dir, seed)

        train_data = env.points[:train_examples]
        train_labels = env.labels[:train_examples]
        val_data = from_numpy(env.points[train_examples: train_examples + eval_examples])
        val_labels = from_numpy(env.labels[train_examples: train_examples + eval_examples])
        test_data = from_numpy(env.points[train_examples + eval_examples: train_examples + eval_examples + test_examples])
        test_labels = from_numpy(env.labels[train_examples + eval_examples: train_examples + eval_examples + test_examples])

        assert train_data.size()[0] == train_examples
        assert val_data.size()[0] == eval_examples
        assert test_data.size()[0] == test_examples

        for i in trange(num_steps, desc="Training", leave=leave_tqdm):
            batch_data = from_numpy(train_data[batch_size * i: batch_size * (i+1)])
            batch_labels = from_numpy(train_labels[batch_size * i: batch_size * (i+1)])

            model.train()
            batch_logits: Tensor = model(batch_data)

            loss: Tensor = loss_fn(batch_logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            with no_grad():
                eval_loss = loss_fn(model(val_data), val_labels)

            self.logger.add_scalar("loss_train", loss, i)
            self.logger.add_scalar("loss_eval", eval_loss, i)

        model.eval()
        with no_grad():
            test_outputs: Tensor = model(test_data)
            test_loss = loss_fn(test_outputs, test_labels)
            test_accuracy = (test_outputs.argmax(dim=1) == test_labels).mean()

        self.logger.add_scalar("loss_test", test_loss, i+1)
        self.logger.add_scalar("accuracy_test", test_accuracy, i+1)

    def benchmark(self) -> None:
        seeds = [41, 42, 43]
        for seed in seeds:
            self.run(seed=seed, leave_tqdm=False, base_log_dir=RESULTS_DIR)

    def get_model(self) -> Module:
        input_dim = 2
        output_dim = 2
        hidden_dim = 64
        n_layers = 2
        nonlinearity_class = ReLU

        in_dims = [input_dim] + [hidden_dim] * n_layers
        out_dims = [hidden_dim] * n_layers + [output_dim]
        activations = [nonlinearity_class] * n_layers + [Identity]

        return Sequential(chain(
            (Linear(i, o), a()) for i, o, a in zip(in_dims, out_dims, activations)
        ))

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def setup_logging(self, base_log_dir: str, seed: int) -> None:
        log_dir = os.path.join(base_log_dir, self.name, f"seed={seed}", f"run{time.time()}")
        self.logger = SummaryWriter(log_dir)
