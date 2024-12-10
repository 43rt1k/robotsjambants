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

""" Run CPG """
import time
import numpy as np
import matplotlib

# adapt as needed for your system
# from sys import platform
# if platform =="darwin":
#   matplotlib.use("Qt5Agg")
# else:
#   matplotlib.use('TkAgg')

from matplotlib import pyplot as plt

from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv


ADD_CARTESIAN_PD = True
TIME_STEP = 0.001
foot_y = 0.0838 # this is the hip length 
sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

env = QuadrupedGymEnv(render=True,              # visualize
                    on_rack=False,              # useful for debugging! 
                    isRLGymInterface=False,     # not using RL
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="TORQUE",
                    add_noise=False,    # start in ideal conditions
                    # record_video=True
                    )

# initialize Hopf Network, supply gait
cpg = HopfNetwork(time_step=TIME_STEP, gait = 'TROT')
#cpg = HopfNetwork(time_step=TIME_STEP, gait = 'BOUND', omega_swing=8*2*np.pi, omega_stance=2*2*np.pi, des_step_len=0.05)

TEST_STEPS = int(10 / (TIME_STEP))
t = np.arange(TEST_STEPS)*TIME_STEP

# [TODO] initialize data structures to save CPG and robot states
q = np.zeros((4, 3, TEST_STEPS))  # Joint positions for each leg and joint
dq = np.zeros((4, 3, TEST_STEPS))  # Joint velocities for each leg and joint
leg_xyz = np.zeros((4, 3, TEST_STEPS))    # Foot positions in Cartesian space
torques = np.zeros((4, 3, TEST_STEPS))           # Torques applied for each leg and joint


############## Sample Gains
# joint PD gains
kp=np.array([100,100,100])
kd=np.array([2,2,2])
# Cartesian PD gains
kpCartesian = np.diag([500]*3)
kdCartesian = np.diag([20]*3)

for j in range(TEST_STEPS):
    # Initialize torque array to send to motors
    action = np.zeros(12) 
    # Get desired foot positions from CPG 
    xs, zs = cpg.update()
    
    # Get current motor angles and velocities
    current_angles = env.robot.GetMotorAngles()  # Rename q to current_angles
    current_velocities = env.robot.GetMotorVelocities()  # Rename dq to current_velocities

    # Loop through desired foot positions and calculate torques
    for i in range(4):
        # Initialize torques for leg i
        tau = np.zeros(3)
        # Get desired foot i pos (xi, yi, zi) in leg frame
        leg_xyz_desired = np.array([xs[i], sideSign[i] * foot_y, zs[i]])
        
        # Call inverse kinematics to get corresponding joint angles
        leg_q = env.robot.ComputeInverseKinematics(i, leg_xyz_desired)
        
        # Joint PD control
        joint_position_error = leg_q - current_angles[3 * i:3 * i + 3]
        leg_dq = np.zeros(3)  # Desired joint velocity
        joint_velocity_error = leg_dq - current_velocities[3 * i:3 * i + 3]
        tau += kp * joint_position_error + kd * joint_velocity_error

        # Cartesian PD control
        if ADD_CARTESIAN_PD:
            J, leg_cur_xyz = env.robot.ComputeJacobianAndPosition(i)
            leg_cur_dq = J @ current_velocities[3 * i:3 * i + 3]
            pos_error = leg_xyz_desired - leg_cur_xyz
            vel_error = -leg_cur_dq
            tau += J.T @ (kpCartesian @ pos_error + kdCartesian @ vel_error)

        # Set tau for leg i in action vector
        action[3 * i:3 * i + 3] = tau

        # Save the current leg's foot position for this timestep
        leg_xyz[i, :, j] = leg_cur_xyz  # Save foot positions in Cartesian space for each leg

    # Send torques to robot and simulate TIME_STEP seconds 
    env.step(action) 

    # Save joint positions and velocities for each timestep
    q[:, :, j] = current_angles.reshape(4, 3)  # Save joint positions
    dq[:, :, j] = current_velocities.reshape(4, 3)  # Save joint velocities
    torques[:, :, j] = tau  # Save torques for each leg





##################################################### 
# PLOTS
#####################################################
# example
# fig = plt.figure()
# plt.plot(t,joint_pos[1,:], label='FR thigh')
# plt.legend()
# plt.show()

# Example plot for joint position of front-right leg thigh joint over time
# plt.figure()
# plt.plot(t, joint_positions[1, 1, :], label='FR thigh position')
# plt.xlabel("Time (s)")
# plt.ylabel("Joint Position (rad)")
# plt.legend()
# plt.show()

# Example plot for foot position in z-axis (height) of front-left leg
# plt.figure()
# plt.plot(t, foot_positions[0, 2, :], label='FL foot z position')
# plt.xlabel("Time (s)")
# plt.ylabel("Foot Z Position (m)")
# plt.legend()
# plt.show()

# Example plot for torque of front-right leg thigh joint over time
# plt.figure()
# plt.plot(t, torques[1, 1, :], label='FR thigh torque')
# plt.xlabel("Time (s)")
# plt.ylabel("Torque (Nm)")
# plt.legend()
# plt.show()