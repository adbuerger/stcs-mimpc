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

import os
import datetime as dt
import numpy as np
import casadi as ca

from system import System

import logging

logger = logging.getLogger(__name__)

class Simulator(System):


    @property
    def time_grid(self):

        return self._timing.time_grid


    @property
    def time_steps(self):

        return self._timing.time_steps


    @property
    def x_data(self):
 
        try:
 
            return np.asarray(self._x_data)
 
        except AttributeError:

            msg = "Simulation results (states) not available yet, call solve() first."
            
            logging.error(msg)
            raise RuntimeError(msg)


    @property
    def u_data(self):

        try:

            return self._u_data

        except AttributeError:

            msg = "Continuous controls not available yet, call solve() first."
            
            logging.error(msg)
            raise RuntimeError(msg)


    @property
    def b_data(self):

        try:

            return self._b_data

        except AttributeError:

            msg = "Binary controls not available yet, call solve() first."
            
            logging.error(msg)
            raise RuntimeError(msg)


    @property
    def c_data(self):

        return self._ambient.c_data


    def _setup_timing(self, timing):

        self._timing = timing


    def _setup_state(self, state):

        self._state = state


    def _setup_ambient(self, ambient):

        self._ambient = ambient


    def _setup_simulator(self):
 
        self._setup_model()
 
        dt = ca.MX.sym("dt")
 
        ode = {"x": self.x, "p": ca.veccat(dt, self.c, self.u, self.b, self.w), \
            "ode": dt * self.f}
 
        self._integrator = ca.integrator("integrator", "cvodes", ode, \
            {"t0": 0.0, "tf": 1.0})
 
        self._remove_unserializable_attributes()     


    def __init__(self, timing, state, ambient):

        super().__init__()

        logger.debug("Initializing simulator ...")

        self._setup_timing(timing=timing)
        self._setup_state(state=state)
        self._setup_ambient(ambient=ambient)
        self._setup_simulator()

        logger.debug("Simulator initialized.")


    def _set_initial_state(self):

        self._x_data = [self._state.x_hat]


    def _setup_controls(self):

        self._u_data = []
        self._b_data = []


    def _initialize_controls(self):

        self._u_data.append(np.zeros(self.nu))
        self._b_data.append(np.zeros(self.nb))

    
    def _set_b_ac(self, pos):

        T_fpsc = self._x_data[-1][self.x_index["T_fpsc"]]
        T_vtsc = self._x_data[-1][self.x_index["T_vtsc"]]

        T_hts = self._x_data[-1][self.x_index["T_hts"][0]]
        T_lts = self._x_data[-1][self.x_index["T_lts"][0]]

        T_amb = self.c_data[pos, self.c_index["T_amb"]]

        if (T_hts > self.p["T_ac_ht_max"]) \
            or (T_hts < self.p["T_ac_ht_min"]) \
            or (T_lts > self.p["T_ac_lt_max"]) \
            or (T_lts < self.p["T_ac_lt_min"]) \
            or ((T_amb + self.p["dT_rc"]) > self.p["T_ac_mt_max"]) \
            or ((T_amb + self.p["dT_rc"]) < self.p["T_ac_mt_min"]):

            self._b_data[-1][self.b_index["b_ac"]] = 0

        elif (T_hts < self.p["T_ac_ht_max"] - self.p_csim["dT_ac_ht"]) \
            or  (T_hts > self.p["T_ac_ht_min"] + self.p_csim["dT_ac_ht"]) \
            or  (T_lts < self.p["T_ac_lt_max"] - self.p_csim["dT_ac_lt"]) \
            or  (T_lts > self.p["T_ac_lt_min"] + self.p_csim["dT_ac_lt"]) \
            or  ((T_amb + self.p["dT_rc"]) < self.p["T_ac_mt_max"] - self.p_csim["dT_ac_mt"]) \
            or  ((T_amb + self.p["dT_rc"]) > self.p["T_ac_mt_min"] + self.p_csim["dT_ac_mt"]):
            
            self._b_data[-1][self.b_index["b_ac"]] = 1


    def _set_b_fc(self, pos):

        T_lts = self._x_data[-1][self.x_index["T_lts"][0]]

        T_amb = self.c_data[pos, self.c_index["T_amb"]]

        if self._b_data[-1][self.b_index["b_ac"]] == 1:

            self._b_data[-1][self.b_index["b_fc"]] = 0

        else:

            if T_lts < (T_amb + self.p["dT_rc"]):

                self._b_data[-1][self.b_index["b_fc"]] = 0 
            
            elif T_lts > (T_amb + 2*self.p["dT_rc"]):

                self._b_data[-1][self.b_index["b_fc"]] = 1 


    def _set_v_ppsc(self):

        T_fpsc = self._x_data[-1][self.x_index["T_fpsc"]]
        T_vtsc = self._x_data[-1][self.x_index["T_vtsc"]]
        T_hts = self._x_data[-1][self.x_index["T_hts"][0]]

        dT = max(T_fpsc, T_vtsc) - T_hts

        v_ppsc = (self.p_op["v_ppsc"]["max"] / (self.p_csim["dT_sc_ub"] - self.p_csim["dT_sc_lb"])) * \
            (max(self.p_csim["dT_sc_lb"], min(self.p_csim["dT_sc_ub"], dT)) - self.p_csim["dT_sc_lb"])

        self._u_data[-1][self.u_index["v_ppsc"]] = v_ppsc


    def _set_p_mpsc(self):

        T_fpsc = self._x_data[-1][self.x_index["T_fpsc"]]
        T_vtsc = self._x_data[-1][self.x_index["T_vtsc"]]

        dT = T_fpsc - T_vtsc

        p_mpsc = ((self.p_op["p_mpsc"]["max"] - self.p_op["p_mpsc"]["min_real"]) / \
            (self.p_csim["dT_vtsc_fpsc_ub"] - self.p_csim["dT_vtsc_fpsc_lb"])) * \
            (max(self.p_csim["dT_vtsc_fpsc_lb"], min(self.p_csim["dT_vtsc_fpsc_ub"], dT)) \
            - self.p_csim["dT_vtsc_fpsc_lb"])

        self._u_data[-1][self.u_index["p_mpsc"]] = p_mpsc
        

    def _set_v_pssc(self):

        T_shx_ssc = self._x_data[-1][self.x_index["T_shx_ssc"][-1]]
        T_hts = self._x_data[-1][self.x_index["T_hts"][0]]

        dT = T_shx_ssc - T_hts

        v_pssc = (self.p_op["v_pssc"]["max"] / (self.p_csim["dT_sc_ub"] - self.p_csim["dT_sc_lb"])) * \
            (max(self.p_csim["dT_sc_lb"], min(self.p_csim["dT_sc_ub"], dT)) - self.p_csim["dT_sc_lb"])

        self._u_data[-1][self.u_index["v_pssc"]] = v_pssc


    def _set_v_plc(self):

        T_r_a = self._x_data[-1][self.x_index["T_r_a"][1]]
        T_r_a_ref = (self.p_op["room"]["T_r_a_max"] - self.p_op["room"]["T_r_a_min"]) / 2

        dT = T_r_a - T_r_a_ref

        v_plc = (self.p_op["v_plc"]["max"] / (self.p_csim["dT_fcu_w_ub"] - self.p_csim["dT_fcu_w_lb"])) * \
            (max(self.p_csim["dT_fcu_w_ub"], min(self.p_csim["dT_fcu_w_ub"], dT)) - self.p_csim["dT_fcu_w_lb"])

        self._u_data[-1][self.u_index["v_plc"]] = v_plc


    def _set_mdot_o_hts_b(self):

        T_shx_ssc = self._x_data[-1][self.x_index["T_shx_ssc"][-1]]
        T_sc_feed_max = self.p_op["T_sc"]["T_feed_max"]

        dT = T_shx_ssc - T_sc_feed_max

        mdot_o_hts_b = (1.0 / (self.p_csim["dT_o_hts_b_ub"] - self.p_csim["dT_o_hts_b_lb"])) * \
            (max(self.p_csim["dT_o_hts_b_lb"], min(self.p_csim["dT_o_hts_b_ub"], dT)) - self.p_csim["dT_o_hts_b_lb"]) * \
                 self.p["mdot_ssc_max"] * self._u_data[-1][self.u_index["v_pssc"]]

        self._u_data[-1][self.u_index["mdot_o_hts_b"]] = mdot_o_hts_b


    def _set_mdot_i_hts_b(self):

        T_hts_m = self._x_data[-1][self.x_index["T_hts"][1]]
        T_i_hts_b_active = self.p_csim["T_i_hts_b_active"]

        dT = T_hts_m - T_i_hts_b_active

        mdot_i_hts_b = (1.0 / (self.p_csim["dT_i_hts_b_ub"] - self.p_csim["dT_i_hts_b_lb"])) * \
            (max(self.p_csim["dT_i_hts_b_lb"], min(self.p_csim["dT_i_hts_b_ub"], dT)) - self.p_csim["dT_i_hts_b_lb"]) * \
                self._b_data[-1][self.b_index["b_ac"]] * self.p["mdot_ac_ht"]

        self._u_data[-1][self.u_index["mdot_i_hts_b"]] = mdot_i_hts_b


    def _set_controls(self, pos):

        self._initialize_controls()

        self._set_b_ac(pos)
        self._set_b_fc(pos)

        self._set_v_ppsc()
        self._set_p_mpsc()
        self._set_v_pssc()
        self._set_v_plc()
        self._set_mdot_o_hts_b()
        self._set_mdot_i_hts_b()


    def _run_step(self, pos, step):

        self._x_data.append( \
            
            self._integrator(x0 = self.x_data[-1], \
                p = ca.veccat( \
                    step,
                    self.c_data[pos],
                    self.u_data[-1],
                    self.b_data[-1],
                    np.zeros(self.nw)))["xf"]
        ) 

    
    def _finalize_simulation_results(self):

        self._x_data = ca.horzcat(*self.x_data).T
        self._u_data = ca.horzcat(*self.u_data).T
        self._b_data = ca.horzcat(*self.b_data).T


    def _run_simulation(self):

        logger.info("Running simulation ...")

        self._set_initial_state()
        self._setup_controls()

        for pos, step in enumerate(self.time_steps):
            
            self._set_controls(pos)
            self._run_step(pos, step)

        self._finalize_simulation_results()

        logger.info("Simulation finished.")


    def solve(self):

        self._run_simulation()


    def save_results(self):

        '''
        This function can be used to save the simulation results, possibly including
        solver runtimes, log files etc.
        '''

        pass

