from torch import Tensor

from awa.modeling.base import ModelOutput


class NoVisMixin:
    def visualize(self, data: Tensor, labels: Tensor) -> None:
        pass

    def setup_visualization_logging(self) -> None:
        pass

    def store_eval_logs_to_visualize(self, output: ModelOutput, loss: Tensor) -> None:
        pass
