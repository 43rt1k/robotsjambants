# SPDX-FileCopyrightText: Copyright (c) 2022 Guillaume Bellegarda. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2022 EPFL, Guillaume Bellegarda

import os, sys
import gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sys import platform
# may be helpful depending on your system
# if platform =="darwin": # mac
#   import PyQt5
#   matplotlib.use("Qt5Agg")
# else: # linux
#   matplotlib.use('TkAgg')

# stable-baselines3
from stable_baselines3.common.monitor import load_results 
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env # fix for newer versions of stable-baselines3

from env.quadruped_gym_env import QuadrupedGymEnv
# utils
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results


LEARNING_ALG = "SAC"
interm_dir = "./logs/intermediate_models/"
# path to saved models, i.e. interm_dir + '120324132026'
log_dir = interm_dir + '120324141512'

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training
env_config = {}
env_config['render'] = True
env_config['record_video'] = False
env_config['add_noise'] = False 
# env_config['competition_env'] = True

# get latest model and normalization stats, and plot 
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
print(monitor_results)
#plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + ' ')
#plt.show() 

# reconstruct env 
env = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
env.training = False    # do not update stats at test time
env.norm_reward = False # reward normalization is not needed at test time

# load model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
print("\nLoaded model", model_name, "\n")

obs = env.reset()
episode_reward = 0

# [TODO] initialize arrays to save data from simulation 
#
velocities = []  # To store the forward velocity (Vx)
heights = []     # To store the height of the robot's base
yaws = []        # To store the yaw angle
drifts = []      # To store the lateral drift (Vy)


for i in range(2000):
    action, _states = model.predict(obs,deterministic=False) # sample at test time? ([TODO]: test)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards
    
    base_velocity = env.envs[0].env.robot.GetBaseLinearVelocity()  # [Vx, Vy, Vz]
    base_position = env.envs[0].env.robot.GetBasePosition()       # [x, y, z]
    base_orientation = env.envs[0].env.robot.GetBaseOrientationRollPitchYaw() # Quaternion [x, y, z, w]
    
    
    if dones:
        print('episode_reward', episode_reward)
        print('Final base position', info[0]['base_pos'])
        episode_reward = 0

    # [TODO] save data from current robot states for plots 
    # To get base position, for example: env.envs[0].env.robot.GetBasePosition() 
    #
    velocities.append(base_velocity[0])  # Forward velocity (Vx)
    heights.append(base_position[2])     # Height (z-coordinate)
    yaws.append(base_orientation[2])     # Assuming yaw is directly available
    drifts.append(base_position[1])      # Lateral velocity (y)
    
# [TODO] make plots:
# Generate time steps for the x-axis
time_steps = range(len(velocities))

# Plot forward velocity (Vx)
plt.figure()
plt.plot(time_steps, velocities, label='Velocity (Vx)')
plt.xlabel("Timestep")
plt.ylabel("Forward Velocity (m/s)")
plt.title("Forward Velocity Over Time")
plt.legend()
plt.grid()
plt.show()

# Plot height (z-coordinate)
plt.figure()
plt.plot(time_steps, heights, label='Height (z)', color='orange')
plt.xlabel("Timestep")
plt.ylabel("Height (m)")
plt.title("Height Over Time")
plt.legend()
plt.grid()
plt.show()

# Plot yaw
plt.figure()
plt.plot(time_steps, yaws, label='Yaw', color='green')
plt.xlabel("Timestep")
plt.ylabel("Yaw (rad)")
plt.title("Yaw Over Time")
plt.legend()
plt.grid()
plt.show()

# Plot lateral drift (Vy)
plt.figure()
plt.plot(time_steps, drifts, label='Drift (Vy)', color='red')
plt.xlabel("Timestep")
plt.ylabel("Drift (m)")
plt.title("Lateral Drift Over Time")
plt.legend()
plt.grid()
plt.show()

# Plot reward evolution
plt.figure()
plt.plot(time_steps, rewards, label='Rewards', color='purple')
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.title("Reward Over Time")
plt.legend()
plt.grid()
plt.show()