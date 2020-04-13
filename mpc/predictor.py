#
# This file is part of stcs-mimpc.
#
# Copyright (c) 2020 Adrian Bürger, Angelika Altmann-Dieses, Moritz Diehl.
# Developed at HS Karlsruhe and IMTEK, University of Freiburg.
# All rights reserved.
#
# The BSD 3-Clause License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
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

import numpy as np
import casadi as ca

from system import System

import logging

logger = logging.getLogger(__name__)

class Predictor(System):

    @property
    def time_grid(self):

        return self._timing.time_grid


    @property
    def x_data(self):

        return self._previous_solver.x_data


    @property
    def x_hat(self):

        try:

            return self._x_hat

        except AttributeError:

            msg = "Estimated states not available yet, call solve() first."

            logging.error(msg)
            raise RuntimeError(msg)


    @property
    def u_data(self):

        return self._previous_solver.u_data


    @property
    def b_data(self):

        return self._previous_solver.b_data


    @property
    def c_data(self):

        return self._ambient.c_data


    @property
    def r_data(self):

        return self._previous_solver.r_data


    @property
    def s_ac_lb_data(self):

        return self._previous_solver.s_ac_lb_data


    @property
    def s_ac_ub_data(self):

        return self._previous_solver.s_ac_ub_data


    @property
    def s_x_data(self):

        return self._previous_solver.s_x_data


    @property
    def s_ppsc_fpsc_data(self):

        return self._previous_solver.s_ppsc_fpsc_data


    @property
    def s_ppsc_vtsc_data(self):

        return self._previous_solver.s_ppsc_vtsc_data


    @property
    def solver_name(self):

        return self._solver_name


    def _setup_timing(self, timing):

        self._timing = timing


    def _setup_state(self, state):

        self._state = state


    def _setup_ambient(self, ambient):

        self._ambient = ambient


    def _set_previous_solver(self, previous_solver):

        self._previous_solver = previous_solver


    def _setup_solver_name(self, solver_name):

        self._solver_name = solver_name


    def _setup_simulator(self):

        self._setup_model()

        dt = ca.MX.sym("dt")

        ode = {"x": self.x, "p": ca.veccat(dt, self.c, self.u, self.b, self.w), \
        "ode": dt * self.f}

        self._integrator = ca.integrator("integrator", "cvodes", ode, \
            {"t0": 0.0, "tf": 1.0})

        self._remove_unserializable_attributes()        


    def __init__(self, timing, state, ambient, previous_solver, solver_name):

        super().__init__()

        self._setup_timing(timing=timing)
        self._setup_state(state=state)
        self._setup_ambient(ambient=ambient)

        self._setup_solver_name(solver_name=solver_name)
        self._set_previous_solver(previous_solver=previous_solver)

        self._setup_simulator()


    def solve(self, n_steps = 1):

        x_hat = self._state.x_hat

        for k in range(n_steps):

            pos_grid = self._timing.grid_position_cursor + k

            x_hat = self._integrator(x0=x_hat, p=ca.veccat( \
                    self._timing.time_steps[pos_grid], self.c_data[pos_grid,:], \
                    self.u_data[pos_grid,:], self.b_data[pos_grid,:], \
                    np.zeros(self.nw)))["xf"]

        self._x_hat = x_hat

