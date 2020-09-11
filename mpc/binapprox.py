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

import time
import datetime as dt
import numpy as np
import casadi as ca
import pycombina

from system import System

import logging

logger = logging.getLogger(__name__)

class BinaryApproximation(System):

    _BINARY_TOLERANCE = 1e-2
    _MAX_NUM_BNB_INTERATIONS = 1e7

    _SECONDS_TIMEOUT_BEFORE_NEXT_TIME_GRID_POINT = 5.0


    @property
    def time_grid(self):

        return self._timing.time_grid


    @property
    def x_data(self):

        return self._previous_solver.x_data


    @property
    def x_hat(self):

        return self._predictor.x_hat


    @property
    def u_data(self):

        try:

            return np.asarray(self._u_data)

        except AttributeError:

            return self._previous_solver.u_data


    @property
    def b_data(self):

        try:

            return self._binapprox.b_bin[:self.nb,:].T

        except AttributeError:

            msg = "Optimized binary controls not available yet, call solve() first."

            logging.error(msg)
            raise RuntimeError(msg)


    @property
    def c_data(self):

        return self._previous_solver.c_data


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

        try:

            return np.asarray(self._s_ppsc_fpsc_data)

        except AttributeError:

            return self._previous_solver.s_ppsc_fpsc_data


    @property
    def s_ppsc_vtsc_data(self):

        try:

            return np.asarray(self._s_ppsc_vtsc_data)

        except AttributeError:

            return self._previous_solver.s_ppsc_vtsc_data


    @property
    def solver_name(self):

        return self._solver_name


    def _setup_timing(self, timing):

        self._timing = timing


    def _set_previous_solver(self, previous_solver):

        self._previous_solver = previous_solver


    def _set_predictor(self, predictor):

        self._predictor = predictor


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


    def _setup_solver_options(self):

        self._solver_options = {}

        self._solver_options["max_iter"] = int(1e9)
        self._solver_options["max_cpu_time"] = 360.0


    def __init__(self, timing, previous_solver, predictor, solver_name):

        super().__init__()

        self._setup_timing(timing = timing)

        self._setup_solver_name(solver_name=solver_name)
        self._set_previous_solver(previous_solver=previous_solver)
        self._set_predictor(predictor=predictor)

        self._setup_simulator()

        self._setup_solver_options()


    def set_solver_max_cpu_time(self, time_point_to_finish):

        if self._timing.time_of_day == "night":
            
            time_for_final_nlp_solve = self._timing.dt_night

        else:
            
            time_for_final_nlp_solve = self._timing.dt_day

        time_for_final_nlp_solve /= self._timing.N_short_term

        max_cpu_time = (time_point_to_finish \
            - dt.datetime.now(tz = self._timing.timezone) \
            - dt.timedelta(seconds = time_for_final_nlp_solve) \
            - dt.timedelta(seconds = self._SECONDS_TIMEOUT_BEFORE_NEXT_TIME_GRID_POINT)).total_seconds()

        self._solver_options["max_cpu_time"] = max_cpu_time

        logger.debug("Maximum CPU time for " + self._solver_name + " set to " + \
            str(max_cpu_time) + " s ...")


    def _setup_binary_approximation_problem(self):

        self._b_rel = np.vstack([np.asarray(self._previous_solver.b_data).T, \
            np.atleast_2d(1-self._previous_solver.b_data.sum(axis=1))]).T

        # Ensure values are not out of range due to numerical effects

        self._b_rel[self._b_rel < 0] = 0
        self._b_rel[self._b_rel > 1.0] = 1

        self._binapprox = pycombina.BinApprox(t=self._timing.time_grid, \
            b_rel=self._b_rel, binary_threshold=self._BINARY_TOLERANCE)


    def _assure_min_up_time_compliance(self):

        if self._timing._remaining_min_up_time > 0:

            logger.debug("Remaining min up time: " + \
                str(self._timing._remaining_min_up_time) + " s")
            logger.debug("The following locking remains: " + \
                str(list(self._timing._b_bin_locked)))

            self._binapprox.set_valid_controls_for_interval( \
                (0.0, self._timing._remaining_min_up_time), \
                np.hstack([self._timing._b_bin_locked, np.zeros((1,1))]))

            logger.debug("Restricted time interval " + \
                str((0.0, self._timing._remaining_min_up_time)) + \
                " to values " + str(self._timing._b_bin_locked) + " s")

        else:

            logger.debug("Setting previously active control: " + \
                str(self._timing._b_bin_prev))

            self._binapprox.set_b_bin_pre(np.hstack( \
                [np.squeeze(self._timing._b_bin_prev), np.zeros((1,))]))        


    def _set_max_number_of_switches(self):

        max_switches = np.around(np.absolute(self._b_rel[1:,:] - self._b_rel[:-1,:]).sum(axis=0))
        max_switches[max_switches < 2] = 2

        max_switches += 1
        max_switches[-1] = np.sum(max_switches[:-1]) + 1

        self._binapprox.set_n_max_switches(n_max_switches=max_switches)


    def _set_min_up_times(self):

        self._binapprox.set_min_up_times(min_up_times= \
            self.p_op["acm"]["min_up_time"] + [0])


    def _set_min_down_times(self):

        self._binapprox.set_min_down_times(min_down_times= \
            self.p_op["acm"]["min_down_time"] + [0])


    def _solve_binary_approximation_problem(self):

        logger.info(self._solver_name + ", iter " + \
            str(self._timing.mpc_iteration_count) + ", limit " + \
            str(self._solver_options["max_cpu_time"]) + " s ...")

        t_start = time.time()

        combina = pycombina.CombinaBnB(self._binapprox)
        combina.solve(**self._solver_options)

        self._runtime = time.time() - t_start

        logger.info(self._solver_name + " finished after " + \
            str(self._runtime) + " s")


    def _update_min_up_times(self):

        starting_point_b_active = 0

        if self._timing._remaining_min_up_time <= 0.0:

            logger.debug("Remaining min up time <= 0 ...")

            if np.all(self.b_data[:self._timing.N_short_term,:self.nb] == 0):

                logger.debug("No active switch, setting remaining min up time = 0.")

                self._timing._remaining_min_up_time = 0.0
                self._timing._b_bin_locked = np.zeros((1, self.nb))

            else:

                logger.debug("Active switch ...")

                b_active = np.where( \
                    self.b_data[:self._timing.N_short_term,:self.nb] == 1)[1][0]
                starting_point_b_active = \
                    np.where(self.b_data[:self._timing.N_short_term,b_active] == 1)[0][0]

                logger.debug("Switch " + str(b_active) + " active on short term time point " + \
                    str(starting_point_b_active) + ".")

                if self.b_data[starting_point_b_active,b_active] != self._timing._b_bin_locked[0,b_active]:

                    logger.debug("Switch differs from previously locked switch ...")

                    self._timing._remaining_min_up_time = \
                        self.p_op["acm"]["min_up_time"][b_active]
                    self._timing._b_bin_locked[:,:] = 0
                    self._timing._b_bin_locked[0,b_active] = 1

                    logger.debug("Setting remaining min up time = " + \
                        str(self._timing._remaining_min_up_time) + \
                        " for control " + str(self._timing._b_bin_locked) + ".")

        logger.debug("Updating min up time ...")

        self._timing._remaining_min_up_time = max(0, \
            self._timing._remaining_min_up_time - self._timing.time_steps[ \
                starting_point_b_active:self._timing.N_short_term].values.sum())

        logger.debug("Setting remaining min up time = " + \
            str(self._timing._remaining_min_up_time))


    def _run_simulation(self):

        x_data = [self._predictor.x_hat]

        for k in range(self._timing.N):

            x_data.append( \
                self._integrator(x0 = x_data[-1], p = ca.veccat( \
                    self._timing.time_steps[k], self.c_data[k,:], \
                    self.u_data[k,:], self.b_data[k,:], np.zeros(self.nw)))["xf"])

        self._x_data = ca.horzcat(*x_data).T


    def _set_feasible_controls_and_slacks(self):

        self._u_data = self._previous_solver.u_data
        self._s_ppsc_fpsc_data = self._previous_solver._s_ppsc_fpsc_data
        self._s_ppsc_vtsc_data = self._previous_solver._s_ppsc_vtsc_data

        idx_T_fpsc_so = np.where(np.squeeze(self.x_data[:,self.x_index["T_fpsc"]] > \
            self.p_op["T_sc"]["T_sc_so"]))[0]
        idx_T_vtsc_so = np.where(np.squeeze(self.x_data[:,self.x_index["T_vtsc"]] > \
            self.p_op["T_sc"]["T_sc_so"]))[0]

        idx_T_sc_so = np.setdiff1d(np.unique( \
            np.concatenate((idx_T_fpsc_so, idx_T_vtsc_so))), self._timing.N)

        self._u_data[idx_T_sc_so, self.u_index["v_ppsc"]] = \
            np.maximum(np.asarray(self._u_data[idx_T_sc_so, self.u_index["v_ppsc"]]), \
                self.p_op["T_sc"]["v_ppsc_so"])

        self._s_ppsc_fpsc_data[idx_T_sc_so] = \
            np.maximum(np.asarray(self._s_ppsc_fpsc_data[idx_T_sc_so]), \
                self.p_op["T_sc"]["v_ppsc_so"])

        self._s_ppsc_vtsc_data[idx_T_sc_so] = \
            np.maximum(np.asarray(self._s_ppsc_vtsc_data[idx_T_sc_so]), \
                self.p_op["T_sc"]["v_ppsc_so"])


    def _improve_initial_guess_for_nlp(self):

        self._run_simulation()
        self._set_feasible_controls_and_slacks()


    def _approximate_binary_controls_from_relaxed_solution(self):

        self._setup_binary_approximation_problem()
        self._assure_min_up_time_compliance()
        self._set_max_number_of_switches()
        self._set_min_up_times()
        self._set_min_down_times()
        self._solve_binary_approximation_problem()
        self._update_min_up_times()
        self._improve_initial_guess_for_nlp()


    def solve(self):

        self._approximate_binary_controls_from_relaxed_solution()


    def save_results(self):

        '''
        This function can be used to save the MPC results, possibly including
        solver runtimes, log files etc.
        '''

        pass

