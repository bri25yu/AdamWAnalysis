from numpy.random import seed

from awa.infra import Env
from awa.vis.env import plot_env


seed(42)
env = Env()
plot_env(env)
