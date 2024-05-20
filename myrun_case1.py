# RL algorithm
import ray
from ray.rllib.algorithms.ppo import PPOConfig
# to use a custom env
from ray.tune.registry import register_env

# my custom env
import numpy as np
from net_env_case1 import NetworkEnv

# Just to suppress
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

ray.init()

# registering my custom env with a name "netenv-v0"
def env_creator(env_config):
    return NetworkEnv()

register_env('netenv-v0', env_creator)


# Set up RL
blconfig = (PPOConfig()
          .training(gamma=0.999, lr=0.0)
          .environment(env='netenv-v0')
          .resources(num_gpus=0)
          .env_runners(num_env_runners=0, num_envs_per_env_runner=1)
        )

baseline = blconfig.build()


for i in range(100):
  res = baseline.train()