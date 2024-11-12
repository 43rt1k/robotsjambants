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

"""
CPG in polar coordinates based on: 
CPG-RL: Learning Central Pattern Generators for Quadruped Locomotion 
authors: Bellegarda, Ijspeert
https://ieeexplore.ieee.org/abstract/document/9932888
"""
import numpy as np

# for RL 
MU_LOW = 1
MU_UPP = 2


class HopfNetwork():
  """ CPG network based on hopf polar equations mapped to foot positions in Cartesian space.  

  Foot Order is FR, FL, RR, RL
  (Front Right, Front Left, Rear Right, Rear Left)
  """
  def __init__(self,
                mu=1**2,                 # intrinsic amplitude, converges to sqrt(mu)
                omega_swing=5*2*np.pi,   # frequency in swing phase (can edit)
                omega_stance=2*2*np.pi,  # frequency in stance phase (can edit)
                gait="TROT",             # Gait, can be TROT, WALK, PACE, BOUND, etc.
                alpha=50,                # amplitude convergence factor
                coupling_strength=1,     # coefficient to multiply coupling matrix
                couple=True,             # whether oscillators should be coupled
                time_step=0.001,         # time step 
                ground_clearance=0.07,   # foot swing height 
                ground_penetration=0.01, # foot stance penetration into ground 
                robot_height=0.3,        # in nominal case (standing) 
                des_step_len=0.05,       # desired step length 
                max_step_len_rl=0.1,     # max step length, for RL scaling 
                use_RL=False             # whether to learn parameters with RL 
                ):
    
    ###############
    # initialize CPG data structures: amplitude is row 0, and phase is row 1
    self.X = np.zeros((2,4))
    self.X_dot = np.zeros((2,4))

    # save parameters 
    self._mu = mu
    self._omega_swing = omega_swing
    self._omega_stance = omega_stance  
    self._alpha = alpha
    self._couple = couple
    self._coupling_strength = coupling_strength
    self._dt = time_step
    self._set_gait(gait)

    # set oscillator initial conditions  
    self.X[0,:] = np.random.rand(4) * .1
    self.X[1,:] = self.PHI[0,:] 

    # save body and foot shaping
    self._ground_clearance = ground_clearance 
    self._ground_penetration = ground_penetration
    self._robot_height = robot_height 
    self._des_step_len = des_step_len

    # for RL
    self.use_RL = use_RL
    self._omega_rl = np.zeros(4)
    self._mu_rl = np.zeros(4) 
    self._max_step_len_rl = max_step_len_rl
    if use_RL:
      self.X[0,:] = MU_LOW # mapping MU_LOW=1 to MU_UPP=2



  def _set_gait(self,gait):
    """ For coupling oscillators in phase space. 
    [TODO] update all coupling matrices
    """
    self.PHI_trot = np.zeros((4,4))
    self.PHI_walk = np.zeros((4,4))
    self.PHI_bound = np.zeros((4,4))
    self.PHI_pace = np.zeros((4,4))
    
    # Coupling matrix for TROT gait: diagonal legs in sync
    self.PHI_trot = np.array([
        [0, np.pi, np.pi, 0],
        [np.pi, 0, 0, np.pi],
        [np.pi, 0, 0, np.pi],
        [0, np.pi, np.pi, 0]
    ])

    # Coupling matrix for PACE gait: left and right legs in sync
    self.PHI_pace = np.array([
        [0, np.pi, 0, np.pi],
        [np.pi, 0, np.pi, 0],
        [0, np.pi, 0, np.pi],
        [np.pi, 0, np.pi, 0]
    ])

    # Coupling matrix for BOUND gait: front and back legs in sync
    self.PHI_bound = np.array([
        [0, 0, np.pi, np.pi],
        [0, 0, np.pi, np.pi],
        [np.pi, np.pi, 0, 0],
        [np.pi, np.pi, 0, 0]
    ])


    # Coupling matrix for WALK gait: sequential with quarter-phase differences
    self.PHI_walk = np.array([
        [0, np.pi, 3*np.pi/2, np.pi/2],  
        [np.pi, 0, np.pi/2, 3*np.pi/2],  
        [np.pi/2, 3*np.pi/2, 0, np.pi],   
        [3*np.pi/2, np.pi/2, np.pi, 0]    
    ])


    if gait == "TROT":
      self.PHI = self.PHI_trot
    elif gait == "PACE":
      self.PHI = self.PHI_pace
    elif gait == "BOUND":
      self.PHI = self.PHI_bound
    elif gait == "WALK":
      self.PHI = self.PHI_walk
    else:
      raise ValueError( gait + 'not implemented.')


  def update(self): 
    """ Update oscillator states. """ 
 
    # update parameters, integrate 
    if not self.use_RL: 
      self._integrate_hopf_equations() 
    else: 
      self._integrate_hopf_equations_rl() 
     
    # map CPG variables to Cartesian foot xz positions (Equations 8, 9)  
    x = np.zeros(4) # [TODO] 
    z = np.zeros(4) # [TODO] 
 
    dstep = self._des_step_len 
    h = self._robot_height 
    gc = self._ground_clearance 
    gp = self._ground_penetration 
    r = self.X[0,:] 
    theta = self.X[1,:] 
 
    x = -dstep*r*np.cos(theta) 
    z = np.where(np.sin(theta) > 0, -h + gc*np.sin(theta), -h + gp*np.sin(theta)) 
 
    # scale x by step length 
    if not self.use_RL: 
      # use des step len, fixed 
      return x, z 
    else: 
      # RL uses amplitude to set max step length 
      r = np.clip(self.X[0,:],MU_LOW,MU_UPP)  
      return -self._max_step_len_rl * (r - MU_LOW) * np.cos(self.X[1,:]), z

      
        
  def _integrate_hopf_equations(self):
    """ 
        Hopf polar equations and integration. Use equations 6 and 7.
        gate =  0(lateral sequence walk) 
                1(trot)  
                2(pace) 
                3(bound)  
    """
    
    # bookkeeping - save copies of current CPG states 
    X = self.X.copy()
    X_dot_prev = self.X_dot.copy() 
    X_dot = np.zeros((2, 4))

    # loop through each leg's oscillator
    for i in range(4):
        # get r_i, theta_i from X
        r, theta = X[:, i]

        # compute r_dot (Equation 6)
        r_dot = self._alpha * (self._mu - r**2) * r

        # determine whether oscillator i is in swing or stance phase
        if 0 <= theta <= np.pi:
            # Swing phase
            omega_i = self._omega_swing
        else:
            # Stance phase
            omega_i = self._omega_stance

        # compute theta_dot (Equation 7)
        theta_dot = omega_i  # natural frequency

        # loop through other oscillators to add coupling (Equation 7)
        if self._couple:
            for j in range(4):
                if j != i:  # Don't couple with itself
                    # coupling strength and phase offset might need to be defined or adjusted
                    theta_dot += r * self._coupling_strength * np.sin(X[1, j] - theta - self.PHI[i, j])

        # set X_dot[:, i]
        X_dot[:, i] = [r_dot, theta_dot]

    # integrate 
    self.X += (X_dot_prev + X_dot) * self._dt / 2  # Semi-implicit Euler method for integration
    self.X_dot = X_dot

    # mod phase variables to keep between 0 and 2pi
    self.X[1, :] = self.X[1, :] % (2 * np.pi)

  ###################### Helper functions for accessing CPG States
  def get_r(self):
    """ Get CPG amplitudes (r) """
    return self.X[0,:]

  def get_theta(self):
    """ Get CPG phases (theta) """
    return self.X[1,:]

  def get_dr(self):
    """ Get CPG amplitude derivatives (r_dot) """
    return self.X_dot[0,:]

  def get_dtheta(self):
    """ Get CPG phase derivatives (theta_dot) """
    return self.X_dot[1,:]

  ###################### Functions for setting parameters for RL
  def set_omega_rl(self, omegas):
    """ Set intrinisc frequencies. """
    self._omega_rl = omegas 

  def set_mu_rl(self, mus):
    """ Set intrinsic amplitude setpoints. """
    self._mu_rl = mus

  def _integrate_hopf_equations_rl(self):
    """ Hopf polar equations and integration, using quantities set by RL """
    # bookkeeping - save copies of current CPG states 
    X = self.X.copy()
    X_dot_prev = self.X_dot.copy() 
    X_dot = np.zeros((2,4))

    # loop through each leg's oscillator, find current velocities
    for i in range(4):
      # get r_i, theta_i from X
      r, theta = X[:,i]
      # amplitude (use mu from RL, i.e. self._mu_rl[i])
      r_dot = 0  # [TODO]
      # phase (use omega from RL, i.e. self._omega_rl[i])
      theta_dot = 0 # [TODO]

      X_dot[:,i] = [r_dot, theta_dot]

    # integrate 
    self.X = X + (X_dot_prev + X_dot) * self._dt / 2
    self.X_dot = X_dot
    self.X[1,:] = self.X[1,:] % (2*np.pi)