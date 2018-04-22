#!/usr/bin/env python

"""
Train an agent on Sonic using PPO2 from OpenAI Baselines.
"""

import tensorflow as tf

from vec_env.dummy_vec_env import DummyVecEnv
from subproc_vec_env import SubprocVecEnv
import baselines.ppo2.ppo2 as ppo2
import baselines.ppo2.policies as policies
# Modified to save and load
import ppo2_modified.ppo2.ppo2 as ppo2
import ppo2_modified.ppo2.policies as policies

import gym_remote.exceptions as gre

from sonic_util import make_env
from tools import getListOfGames


def getEnvs():
    envs = [make_env for i in range(0, len(getListOfGames("train")))]
    print(envs, "******************* ENVS BEFORE SubprocVecEnv **************************")
    envs = SubprocVecEnv(envs)
    print(envs, "******************* ENVS AFTER SubprocVecEnv **************************")
    return envs

def main():
    """Run PPO until the environment throws an exception."""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    with tf.Session(config=config):
        # Take more timesteps than we need to be sure that
        # we stop due to an exception.
        envs = getEnvs()

        ppo2.learn(policy=policies.CnnPolicy,
                   env=envs,
                   nsteps=4096,
                   nminibatches=8,
                   lam=0.95,
                   gamma=0.99,
                   noptepochs=3,
                   log_interval=1,
                   ent_coef=0.01,
                   lr=lambda _: 2e-4,
                   cliprange=lambda _: 0.1,
                   total_timesteps=int(5*1e7),
                   save_interval=1e7)

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
