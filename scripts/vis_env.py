from numpy.random import seed

from awa.infra import Env
from awa.vis.env import plot_env


seed(42)
env = Env(N=100000, C=2, D=2)
plot_env(env)
