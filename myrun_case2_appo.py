# RL algorithm
import ray
from ray.rllib.algorithms.appo import APPOConfig
# to use a custom env
from ray.tune.registry import register_env
from ray.rllib.algorithms.impala import ImpalaConfig
# my custom env
import numpy as np
from net_env_case2 import NetworkEnv

# Just to suppress
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

ray.init()

# registering my custom env with a name "netenv-v0"
def env_creator(env_config):
    return NetworkEnv()

register_env('netenv-v0', env_creator)
# Set up RL
config2 = (
    APPOConfig()
    .training(lr=0.001, grad_clip=30.0, train_batch_size=50)
    .environment(env="netenv-v0")
    .resources(num_gpus=0)
    .env_runners(num_env_runners=0, num_envs_per_env_runner=1)
)

algo = config2.build()

for i in range(800):
  res = algo.train()