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
import copy
import datetime as dt
import numpy as np
import casadi as ca

from system import System
from nlpsetup import NLPSetupMPC, NLPSetupMHE

from abc import ABCMeta, abstractmethod

import logging

logger = logging.getLogger(__name__)

class NLPSolverMPCBaseClass(NLPSetupMPC, metaclass = ABCMeta):

    _SECONDS_TIMEOUT_BEFORE_NEXT_TIME_GRID_POINT = 5.0
    _LOGFILE_LOCATION = "/tmp"


    @property
    def time_grid(self):

        return self._timing.time_grid


    @property
    def x_data(self):

        try:

            return np.asarray(self._x_data)

        except AttributeError:

            msg = "Optimized states not available yet, call solve() first."

            logging.error(msg)
            raise RuntimeError(msg)


    @property
    def x_hat(self):

        return np.asarray(self._predictor.x_hat)


    @property
    def u_data(self):

        try:

            return np.asarray(self._u_data)

        except AttributeError:

            msg = "Optimized continuous controls not available yet, call solve() first."

            logging.error(msg)
            raise RuntimeError(msg)


    @property
    def b_data(self):

        try:

            return np.asarray(self._b_data)

        except AttributeError:

            msg = "Optimized binary controls not available yet, call solve() first."

            logging.error(msg)
            raise RuntimeError(msg)


    @property
    def c_data(self):

        return np.asarray(self._ambient.c_data)


    @property
    def r_data(self):

        try:

            return np.asarray(self._r_data)

        except AttributeError:

            msg = "Optimized residuals not available yet, call solve() first."

            logging.error(msg)
            raise RuntimeError(msg)


    @property
    def s_ac_lb_data(self):

        try:

            return np.asarray(self._s_ac_lb_data)

        except AttributeError:

            msg = "Optimized slacks for minimum AC operation temperatures not available yet, call solve() first."

            logging.error(msg)
            raise RuntimeError(msg)


    @property
    def s_ac_ub_data(self):

        try:

            return np.asarray(self._s_ac_ub_data)

        except AttributeError:

            msg = "Optimized slacks for maximum AC operation temperatures not available yet, call solve() first."

            logging.error(msg)
            raise RuntimeError(msg)


    @property
    def s_x_data(self):

        try:

            return np.asarray(self._s_x_data)

        except AttributeError:

            msg = "Optimized slacks for soft state constraints not available yet, call solve() first."

            logging.error(msg)
            raise RuntimeError(msg)


    @property
    def s_ppsc_fpsc_data(self):

        try:

            return np.asarray(self._s_ppsc_fpsc_data)

        except AttributeError:

            msg = "Optimized slacks for FPSC safety pump speed not available yet, call solve() first."

            logging.error(msg)
            raise RuntimeError(msg)


    @property
    def s_ppsc_vtsc_data(self):

        try:

            return np.asarray(self._s_ppsc_vtsc_data)

        except AttributeError:

            msg = "Optimized slacks for VTSC safety pump speed not available yet, call solve() first."

            logging.error(msg)
            raise RuntimeError(msg)


    @property
    def solver_name(self):

        return self._solver_name


    def _setup_timing(self, timing):

        self._timing = timing


    def _setup_ambient(self, ambient):

        self._ambient = ambient


    def _set_previous_solver(self, previous_solver):

        self._previous_solver = previous_solver


    def _set_predictor(self, predictor):

        self._predictor = predictor


    def _setup_solver_name(self, solver_name):

        self._solver_name = solver_name


    def _setup_general_solver_options(self):

        self._nlpsolver_options = {}

        self._nlpsolver_options["ipopt.linear_solver"] = "mumps"

        self._nlpsolver_options["ipopt.mumps_mem_percent"] = 10000
        self._nlpsolver_options["ipopt.mumps_pivtol"] = 0.001

        self._nlpsolver_options["ipopt.print_level"] = 5

        self._nlpsolver_options["ipopt.output_file"] = os.path.join( \
            self._LOGFILE_LOCATION, self._solver_name + ".log")
        self._nlpsolver_options["ipopt.file_print_level"] = 5

        self._nlpsolver_options["ipopt.max_cpu_time"] = 720.0


    @abstractmethod
    def _setup_additional_nlpsolver_options(self):

        pass


    def __init__(self, timing, ambient, previous_solver, predictor, solver_name):

        logger.debug("Initializing NLP solver " + solver_name + " ...")

        super().__init__(timing)

        self._setup_timing(timing = timing)
        self._setup_ambient(ambient = ambient)
        
        self._setup_solver_name(solver_name = solver_name)
        self._set_previous_solver(previous_solver = previous_solver)
        self._set_predictor(predictor = predictor)

        self._setup_general_solver_options()
        self._setup_additional_nlpsolver_options()
        self._setup_collocation_options()

        logger.debug("NLP solver "  + solver_name + " initialized.")


    def set_solver_max_cpu_time(self, time_point_to_finish):

        max_cpu_time = (time_point_to_finish - dt.datetime.now(tz = self._timing.timezone) - \
            dt.timedelta(seconds = self._SECONDS_TIMEOUT_BEFORE_NEXT_TIME_GRID_POINT)).total_seconds()

        self._nlpsolver_options["ipopt.max_cpu_time"] = max_cpu_time

        logger.debug("Maximum CPU time for " + self._solver_name + " set to " + \
            str(max_cpu_time) + " s ...")


    def _setup_nlpsolver(self):

        __dirname__ = os.path.dirname(os.path.abspath(__file__))
        path_to_nlp_object = os.path.join(__dirname__, self._PATH_TO_NLP_OBJECT, \
            self._NLP_OBJECT_FILENAME)

        self._nlpsolver = ca.nlpsol(self._solver_name, "ipopt", path_to_nlp_object, 
            self._nlpsolver_options)


    def _set_states_bounds(self):

        '''
        The boundary values for the states will later be defined as soft constraints.
        '''

        self.x_min = self.p_op["T"]["min"] * np.ones( \
            (self._timing.N+1, self.nx - self.nx_aux))
        self.x_max = self.p_op["T"]["max"] * np.ones( \
            (self._timing.N+1, self.nx - self.nx_aux))

        self.x_max[:,self.x_index["T_shx_psc"][-1]] = \
            self.p_op["T_sc"]["T_feed_max"]


    def _set_continuous_control_bounds(self):

        self.u_min = np.hstack([

            self.p_op["v_ppsc"]["min_mpc"] * np.ones((self._timing.N, 1)),
            self.p_op["p_mpsc"]["min_mpc"] * np.ones((self._timing.N, 1)),
            self.p_op["v_plc"]["min_mpc"] * np.ones((self._timing.N, 1)),
            self.p_op["v_pssc"]["min_mpc"] * np.ones((self._timing.N, 1)),

            # The upcoming controls are constrained later in the NLP using inequality constraints

            np.zeros((self._timing.N, 1)),
            np.zeros((self._timing.N, 1))])
        
        self.u_max = np.hstack([
        
            self.p_op["v_ppsc"]["max"] * np.ones((self._timing.N, 1)),
            self.p_op["p_mpsc"]["max"] * np.ones((self._timing.N, 1)),
            self.p_op["v_plc"]["max"] * np.ones((self._timing.N, 1)),
            self.p_op["v_pssc"]["max"] * np.ones((self._timing.N, 1)),

            np.inf * np.ones((self._timing.N, 1)),
            np.inf * np.ones((self._timing.N, 1))])


    @abstractmethod
    def _set_binary_control_bounds(self):

        pass


    def _set_nlpsolver_bounds_and_initials(self):

        # Optimization variables bounds and initials

        V_min = []
        V_max = []
        V_init = []

        # Constraints bounds

        lbg = []
        ubg = []

        # Time-varying parameters

        P_data = []

        # Initial states

        if self._timing.grid_position_cursor == 0:

            V_min.append(self._predictor.x_hat)
            V_max.append(self._predictor.x_hat)
            V_init.append(self._predictor.x_hat)

        else:

            V_min.append(self._previous_solver.x_data[0,:])
            V_max.append(self._previous_solver.x_data[0,:])
            V_init.append(self._previous_solver.x_data[0,:])         


        for k in range(self._timing.N):


            # Collocation equations

            for j in range(1,self.d+1):

                lbg.append(np.zeros(self.nx))
                ubg.append(np.zeros(self.nx))

            if k < self._timing.grid_position_cursor:

                lbg.append(-np.inf * np.ones(self.nx))
                ubg.append(np.inf * np.ones(self.nx))

            else:

                lbg.append(np.zeros(self.nx))
                ubg.append(np.zeros(self.nx))

            # s_ac_lb

            lbg.append(-1e-1 * np.ones(3)) # vanishing constraints smoothened
            ubg.append(np.inf * np.ones(3))

            # s_ac_ub

            lbg.append(-np.inf * np.ones(3))
            ubg.append(1e-1 * np.ones(3)) # vanishing constraints smoothened

            # Setup objective temperature range condition

            lbg.append(self.p_op["room"]["T_r_a_min"])
            ubg.append(self.p_op["room"]["T_r_a_max"])

            # State limits soft constraints

            lbg.append(self.x_min[k+1,:])
            ubg.append(self.x_max[k+1,:])

            # Assure ppsc is running at high speed when collector temperature is high

            lbg.append(-np.inf * np.ones(4))
            ubg.append(1.0e-1)
            ubg.append(0)
            ubg.append(1.0e-1)
            ubg.append(0)

            # Assure HTS bottom layer mass flows are always smaller or equal to
            # the corresponding total pump flow

            lbg.append(-np.inf * np.ones(2))
            ubg.append(np.zeros(2))

            # SOS1 constraints

            lbg.append(0)
            ubg.append(1)

            # Append new bounds and initials

            for j in range(1,self.d+1):

                V_min.append(-np.inf * np.ones(self.nx))
                V_max.append(np.inf * np.ones(self.nx))
                V_init.append(self._previous_solver.x_data[k,:])


            if  k < self._timing.grid_position_cursor:

                V_min.append(self._previous_solver.b_data[k,:])
                V_max.append(self._previous_solver.b_data[k,:])
                V_init.append(self._previous_solver.b_data[k,:])

            else:

                V_min.append(self.b_min[k,:])
                V_max.append(self.b_max[k,:])
                V_init.append(self._previous_solver.b_data[k,:])

            V_min.append(np.zeros(3))
            V_max.append(np.inf * np.ones(3))
            try:
                V_init.append(self._previous_solver.s_ac_lb_data[k,:])
            except AttributeError:
                V_init.append(np.zeros(3))

            V_min.append(np.zeros(3))
            V_max.append(np.inf * np.ones(3))
            try:
                V_init.append(self._previous_solver.s_ac_ub_data[k,:])
            except AttributeError:
                V_init.append(np.zeros(3))

            V_min.append(-np.inf * np.ones(self.nx-self.nx_aux))
            V_max.append(np.inf * np.ones(self.nx-self.nx_aux))
            try:
                V_init.append(self._previous_solver.s_x_data[k,:])
            except AttributeError:
                V_init.append(np.zeros(self.nx-self.nx_aux))

            V_min.append(np.zeros(2))
            V_max.append(self.p_op["v_ppsc"]["max"] * np.ones(2))
            try:
                V_init.append(self._previous_solver.s_ppsc_fpsc_data[k,:])
            except AttributeError:
                V_init.append(0.0)
            try:
                V_init.append(self._previous_solver.s_ppsc_vtsc_data[k,:])
            except AttributeError:
                V_init.append(0.0)

            if k < self._timing.grid_position_cursor:

                V_min.append(self._previous_solver.u_data[k,:])
                V_max.append(self._previous_solver.u_data[k,:])
                V_init.append(self._previous_solver.u_data[k,:])

            else:

                V_min.append(self.u_min[k,:])
                V_max.append(self.u_max[k,:])
                V_init.append(self._previous_solver.u_data[k,:])


            V_min.append(-np.inf)
            V_max.append(np.inf)

            try:
                V_init.append(self._previous_solver.r_data[k,:])
            except AttributeError:
                V_init.append(0.0)

            if (k+1) == self._timing.grid_position_cursor:

                V_min.append(self._predictor.x_hat)
                V_max.append(self._predictor.x_hat)
                V_init.append(self._predictor.x_hat)                

            elif (k+1) < self._timing.grid_position_cursor:

                V_min.append(self._previous_solver.x_data[k+1,:])
                V_max.append(self._previous_solver.x_data[k+1,:])
                V_init.append(self._previous_solver.x_data[k+1,:])

            else:

                V_min.append(-np.inf * np.ones(self.nx))
                V_max.append(np.inf * np.ones(self.nx))
                V_init.append(self._previous_solver.x_data[k+1,:])

            # Append time-varying parameters

            P_data.append(self._ambient.c_data[k,:])
            P_data.append(self._timing.time_steps[k])
            P_data.append(np.zeros(self.nw))


        # Concatenate objects

        self.V_min = ca.veccat(*V_min)
        self.V_max = ca.veccat(*V_max)
        self.V_init = ca.veccat(*V_init)

        self.lbg = np.hstack(lbg)
        self.ubg = np.hstack(ubg)

        self.P_data = ca.veccat(*P_data)

        self._nlpsolver_args = {"p": self.P_data, \
            "x0": self.V_init,
            "lbx": self.V_min, "ubx": self.V_max, \
            "lbg": self.lbg, "ubg": self.ubg}


    def _run_nlpsolver(self):

        logger.info(self._solver_name + ", iter " + \
            str(self._timing.mpc_iteration_count) + ", limit " + \
            str(round(self._nlpsolver_options["ipopt.max_cpu_time"],1)) + " s ...")

        self.nlp_solution = self._nlpsolver(**self._nlpsolver_args)

        if self._nlpsolver.stats()["return_status"] == "Maximum_CpuTime_Exceeded":

            logger.warning(self._solver_name + " returned '" + \
               str(self._nlpsolver.stats()["return_status"]) + "' after " + \
                str(round(self._nlpsolver.stats()["t_wall_total"], 2)) + " s")

        else:

            logger.info(self._solver_name + " returned '" + \
               str(self._nlpsolver.stats()["return_status"]) + "' after " + \
                str(round(self._nlpsolver.stats()["t_wall_total"], 2)) + " s")


    def _collect_nlp_results(self):

        v_opt = np.array(self.nlp_solution["x"])

        x_opt = []
        u_opt = []
        b_opt = []
        r_opt = []
        s_ac_lb_opt = []
        s_ac_ub_opt = []
        s_x_opt = []
        s_ppsc_fpsc_opt = []
        s_ppsc_vtsc_opt = []

        offset = 0

        for k in range(self._timing.N):

            x_opt.append(v_opt[offset:offset+self.nx])

            for j in range(self.d+1):

                offset += self.nx

            b_opt.append(v_opt[offset:offset+self.nb])
            offset += self.nb

            s_ac_lb_opt.append(v_opt[offset:offset+3])
            offset += 3

            s_ac_ub_opt.append(v_opt[offset:offset+3])
            offset += 3

            s_x_opt.append(v_opt[offset:offset+self.nx-self.nx_aux])
            offset += self.nx-self.nx_aux

            s_ppsc_fpsc_opt.append(v_opt[offset:offset+1])
            offset += 1

            s_ppsc_vtsc_opt.append(v_opt[offset:offset+1])
            offset += 1

            u_opt.append(v_opt[offset:offset+self.nu])
            offset += self.nu

            r_opt.append(v_opt[offset:offset+1])
            offset += 1

        x_opt.append(v_opt[offset:offset+self.nx])
        offset += self.nx

        r_opt.append(v_opt[offset:offset+1])
        offset += 1

        self._x_data = ca.horzcat(*x_opt).T
        self._u_data = ca.horzcat(*u_opt).T
        self._b_data = ca.horzcat(*b_opt).T

        self._r_data = ca.horzcat(*r_opt).T

        self._s_ac_lb_data = ca.horzcat(*s_ac_lb_opt).T
        self._s_ac_ub_data = ca.horzcat(*s_ac_ub_opt).T
        self._s_x_data = ca.horzcat(*s_x_opt).T
        self._s_ppsc_fpsc_data = ca.horzcat(*s_ppsc_fpsc_opt).T
        self._s_ppsc_vtsc_data = ca.horzcat(*s_ppsc_vtsc_opt).T


    def solve(self):

        self._setup_nlpsolver()
        self._set_states_bounds()
        self._set_continuous_control_bounds()
        self._set_binary_control_bounds()
        self._set_nlpsolver_bounds_and_initials()
        self._run_nlpsolver()
        self._collect_nlp_results()


    def reduce_object_memory_size(self):

        self._previous_solver = None
        self._predictor = None


    def save_results(self):

        '''
        This function can be used to save the MPC results, possibly including
        solver runtimes, log files etc.
        '''

        pass


class NLPSolverBin(NLPSolverMPCBaseClass):


    def _setup_additional_nlpsolver_options(self):

        self._nlpsolver_options["ipopt.acceptable_tol"] = 0.2
        self._nlpsolver_options["ipopt.acceptable_iter"] = 8
        self._nlpsolver_options["ipopt.acceptable_constr_viol_tol"] = 10.0
        self._nlpsolver_options["ipopt.acceptable_dual_inf_tol"] = 10.0
        self._nlpsolver_options["ipopt.acceptable_compl_inf_tol"] = 10.0
        self._nlpsolver_options["ipopt.acceptable_obj_change_tol"] = 1e-1
        
        self._nlpsolver_options["ipopt.mu_strategy"] = "adaptive"
        self._nlpsolver_options["ipopt.mu_target"] = 1e-5


    def _set_binary_control_bounds(self):

        self.b_min = self._previous_solver.b_data
        self.b_max = self._previous_solver.b_data


class NLPSolverRel(NLPSolverMPCBaseClass):


    def _setup_additional_nlpsolver_options(self):

        self._nlpsolver_options["ipopt.acceptable_tol"] = 0.2
        self._nlpsolver_options["ipopt.acceptable_iter"] = 8
        self._nlpsolver_options["ipopt.acceptable_constr_viol_tol"] = 10.0
        self._nlpsolver_options["ipopt.acceptable_dual_inf_tol"] = 10.0
        self._nlpsolver_options["ipopt.acceptable_compl_inf_tol"] = 10.0
        self._nlpsolver_options["ipopt.acceptable_obj_change_tol"] = 1e-1

        self._nlpsolver_options["ipopt.mu_strategy"] = "adaptive"
        self._nlpsolver_options["ipopt.mu_target"] = 1e-5


    def _set_binary_control_bounds(self):

        self.b_min = np.zeros((self._timing.N, self.nb))
        self.b_max = np.ones((self._timing.N,self.nb))
        self.b_max[:,-1] = 0.0

        if self._timing._remaining_min_up_time > 0:

            locked_time_grid_points = np.where(self._timing.time_grid < self._timing._remaining_min_up_time)[0]

            self.b_min[locked_time_grid_points, :] = np.repeat(self._timing._b_bin_locked, len(locked_time_grid_points), 0)
            self.b_max[locked_time_grid_points, :] = np.repeat(self._timing._b_bin_locked, len(locked_time_grid_points), 0)
            

class NLPSolverMHE(NLPSetupMHE):

    @property
    def x_data(self):

        try:

            return np.asarray(self._x_data)

        except AttributeError:

            msg = "Optimized states not available yet, call solve() first."

            logging.error(msg)
            raise RuntimeError(msg)


    @property
    def x_hat(self):

        try:

            return np.squeeze(self._x_data[-1,:])

        except AttributeError:

            msg = "Current states estimate not available yet, call solve() first."

            logging.error(msg)
            raise RuntimeError(msg)


    @property
    def w_data(self):

        try:

            return np.asarray(self._w_data)

        except AttributeError:

            msg = "Optimized process noise not available yet, call solve() first."

            logging.error(msg)
            raise RuntimeError(msg)


    @property
    def u_data(self):

        return self._measurement.u_data


    @property
    def b_data(self):

        return self._measurement.b_data


    @property
    def c_data(self):

        return self._ambient.c_data


    @property
    def solver_name(self):

        return self._solver_name


    def _setup_ambient(self, ambient):

        self._ambient = ambient


    def _setup_measurement(self, measurement):

        self._measurement = measurement


    def _setup_solver_name(self, solver_name):

        self._solver_name = solver_name


    def _setup_solver_options(self):

        self._nlpsolver_options = {}
        self._nlpsolver_options["ipopt.linear_solver"] = "mumps"
        self._nlpsolver_options["ipopt.print_level"] = 5
        self._nlpsolver_options["ipopt.max_cpu_time"] = 1.0e6


    def _initialize_arrival_cost_covariance(self):

        self.P_x_arr = copy.deepcopy(self.R_w)


    def _initialize_arrival_cost(self):

        # Initialize arrival cost with initial measurement

        self.x_arr = np.zeros(self.nx)

        self.x_arr[self.x_index["T_hts"][0]] = \
            self._measurement.y_data[0, self.y_index["T_hts"][0]]

        self.x_arr[self.x_index["T_hts"][1]] = \
            self._measurement.y_data[0, self.y_index["T_hts"][1]]
        
        self.x_arr[self.x_index["T_hts"][2]] = \
            self._measurement.y_data[0, self.y_index["T_hts"][2]]

        self.x_arr[self.x_index["T_hts"][3]] = \
            self._measurement.y_data[0, self.y_index["T_hts"][3]]

        self.x_arr[self.x_index["T_lts"][0]] = \
            self._measurement.y_data[0, self.y_index["T_lts"][0]]

        self.x_arr[self.x_index["T_lts"][1]] = \
            self._measurement.y_data[0, self.y_index["T_lts"][1]]

        self.x_arr[self.x_index["T_fpsc"]] = \
            self._measurement.y_data[0, self.y_index["T_fpsc_s"]]
        self.x_arr[self.x_index["T_fpsc_s"]] = \
            self._measurement.y_data[0, self.y_index["T_fpsc_s"]]

        self.x_arr[self.x_index["T_vtsc"]] = \
            self._measurement.y_data[0, self.y_index["T_vtsc_s"]]
        self.x_arr[self.x_index["T_vtsc_s"]] = \
            self._measurement.y_data[0, self.y_index["T_vtsc_s"]]

        self.x_arr[self.x_index["T_pscf"]] = \
            self._measurement.y_data[0, self.y_index["T_shx_psc"][1]]
        self.x_arr[self.x_index["T_pscr"]] = \
            self._measurement.y_data[0, self.y_index["T_shx_psc"][0]]

        self.x_arr[self.x_index["T_shx_psc"][:2]] = \
            self._measurement.y_data[0, self.y_index["T_shx_psc"][0]]
        self.x_arr[self.x_index["T_shx_psc"][2:]] = \
            self._measurement.y_data[0, self.y_index["T_shx_psc"][1]]
        self.x_arr[self.x_index["T_shx_ssc"][:2]] = \
            self._measurement.y_data[0, self.y_index["T_shx_ssc"][0]]
        self.x_arr[self.x_index["T_shx_ssc"][2:]] = \
            self._measurement.y_data[0, self.y_index["T_shx_ssc"][1]]

        self.x_arr[self.x_index["T_fcu_w"]] = \
            self._measurement.y_data[0, self.y_index["T_fcu_w"]]
        self.x_arr[self.x_index["T_fcu_a"]] = \
            self._measurement.y_data[0, self.y_index["T_r_a"][0]]

        self.x_arr[self.x_index["T_r_c"]] = \
            self._measurement.y_data[0, self.y_index["T_r_c"]]
        self.x_arr[self.x_index["T_r_a"]] = \
            self._measurement.y_data[0, self.y_index["T_r_a"]]


    def _initialize_ekf_for_arrival_cost_update(self):

        self.hfcn = ca.Function("h", [self.x, self.c], [self.h])
        self.H = self.hfcn.jac()

        ode = {"x": self.x, "p": ca.veccat(self.c, self.u, self.b, self.w), \
            "ode": self.f}

        self.phi = ca.integrator("integrator", "cvodes", ode, \
            {"t0": 0.0, "tf": self._timing.dt_day})
        self.Phi = self.phi.jac()


    def __init__(self, timing, ambient, measurement, solver_name):

        logger.info("Initializing NLP solver " + solver_name + " ...")

        super().__init__(timing)

        self._setup_ambient(ambient = ambient)
        self._setup_measurement(measurement = measurement)
        
        self._setup_solver_name(solver_name = solver_name)

        self._setup_solver_options()
        self._setup_collocation_options()

        self._setup_model()
        self._initialize_arrival_cost_covariance()
        self._initialize_arrival_cost()
        self._initialize_ekf_for_arrival_cost_update()

        logger.info("NLP solver "  + solver_name + " initialized.")


    def _setup_nlpsolver(self):

        __dirname__ = os.path.dirname(os.path.abspath(__file__))
        path_to_nlp_object = os.path.join(__dirname__, self._PATH_TO_NLP_OBJECT, \
            self._NLP_OBJECT_FILENAME)

        self._nlpsolver = ca.nlpsol(self._solver_name, "ipopt", path_to_nlp_object, 
            self._nlpsolver_options)


    def _set_nlpsolver_bounds_and_initials(self):

        # Optimization variables bounds and initials

        V_min = []
        V_max = []
        V_init = []

        # Constraints bounds

        lbg = []
        ubg = []

        # Time-varying parameters

        P_data = []

        # Initial states

        V_min.append(-np.inf * np.ones(self.nx-2))
        V_min.append(np.zeros(2))
        V_max.append(np.inf * np.ones(self.nx-2))
        V_max.append(2.5 * np.ones(2))
        try:
            V_init.append(self._x_data[0,:])
        except AttributeError:
            V_init.append(self.x_arr)

        P_data.append(self.x_arr)
        P_data.append(np.linalg.inv(self.P_x_arr))


        for k in range(self._timing.N):

            # Collocation equations

            for j in range(1,self.d+1):

                lbg.append(np.zeros(self.nx))
                ubg.append(np.zeros(self.nx))

            lbg.append(np.zeros(self.nx))
            ubg.append(np.zeros(self.nx))

            # Append new boundaries and initials

            for j in range(1,self.d+1):

                V_min.append(-np.inf * np.ones(self.nx))
                V_max.append(np.inf * np.ones(self.nx))
                try:
                    V_init.append(self._x_data[k,:])
                except AttributeError:
                    V_init.append(self.x_arr)


            P_data.append(self._measurement.b_data[k,:])
            P_data.append(self._measurement.u_data[k,:])
            P_data.append(self._measurement.y_data[k,:])
            P_data.append(self._ambient.c_data[k,:])
            P_data.append(self._timing.time_steps[k])

            V_min.append(-np.inf * np.ones(self.nw))
            V_max.append(np.inf * np.ones(self.nw))
            V_init.append(np.zeros(self.nw))

            V_min.append(-np.inf * np.ones(self.nx-2))
            V_min.append(np.zeros(2))
            V_max.append(np.inf * np.ones(self.nx-2))
            V_max.append(1.0 * np.ones(2))
            try:
                V_init.append(self._x_data[k+1,:])
            except AttributeError:
                V_init.append(self.x_arr)

        # Append time-varying parameters

        P_data.append(self._measurement.y_data[k+1,:])


        # Concatenate objects

        self.V_min = ca.veccat(*V_min)
        self.V_max = ca.veccat(*V_max)
        self.V_init = ca.veccat(*V_init)

        self.lbg = np.hstack(lbg)
        self.ubg = np.hstack(ubg)

        self.P_data = ca.veccat(*P_data)

        self._nlpsolver_args = {"p": self.P_data, \
            "x0": self.V_init,
            "lbx": self.V_min, "ubx": self.V_max, \
            "lbg": self.lbg, "ubg": self.ubg}


    def _run_nlpsolver(self):

        logger.info("Running " + self._solver_name + " at MPC iteration " + \
            str(self._timing.mhe_iteration_count) + " ...")

        self.nlp_solution = self._nlpsolver(**self._nlpsolver_args)

        logger.info("Solver " + self._solver_name + " finished with '" + \
           str(self._nlpsolver.stats()["return_status"]) + "' after " + \
            str(round(self._nlpsolver.stats()["t_wall_total"], 2)) + " s ...")


    def _collect_nlp_results(self):

        v_opt = np.array(self.nlp_solution["x"])

        x_opt = []
        w_opt = []

        offset = 0

        for k in range(self._timing.N):

            x_opt.append(v_opt[offset:offset+self.nx])

            for j in range(self.d+1):

                offset += self.nx

            w_opt.append(v_opt[offset:offset+self.nw])
            offset += self.nw

        x_opt.append(v_opt[offset:offset+self.nx])
        offset += self.nx

        self._x_data = ca.horzcat(*x_opt).T
        self._w_data = ca.horzcat(*w_opt).T


    def _update_arrival_cost(self):

        # cf. CasADi example mhe_spring_damper.py
        
        H0 = self.H(i0=self.x_data[0,:], i1=self._ambient.c_data[0,:])["Do0Di0"]
        K = ca.mtimes([self.P_x_arr, H0.T, \
            np.linalg.inv(ca.mtimes([H0, self.P_x_arr, H0.T]) + self.W_r)])
        self.P_x_arr = ca.mtimes((ca.DM.eye(self.nx) - ca.mtimes(K, H0)), self.P_x_arr)

        h0 = self.hfcn(self.x_data[0,:], self._ambient.c_data[0,:])
        self.x_arr = np.squeeze(self.x_arr) + ca.mtimes(K, \
            self._measurement.y_data[0,:] - h0 - ca.mtimes(H0, np.squeeze(self.x_arr) - self.x_data[0,:]))
        self.x_arr = np.asarray(self.phi(x0 = self.x_arr, \
            p = ca.veccat(self._ambient.c_data[0,:], self._measurement.u_data[0,:], \
                self._measurement.b_data[0,:], self.w_data[0,:]))["xf"])

        F = self.Phi(x0 = self.x_data[0,:], \
            p = ca.veccat(self._ambient.c_data[0,:], self._measurement.u_data[0,:], \
                self._measurement.b_data[0,:], self.w_data[0,:]))["DxfDx0"]
        self.P_x_arr = np.asarray(ca.mtimes([F, self.P_x_arr, F.T])) + self.R_w


    def solve(self):

        self._setup_nlpsolver()
        self._set_nlpsolver_bounds_and_initials()
        self._run_nlpsolver()
        self._collect_nlp_results()
        self._update_arrival_cost()


    def save_results(self):

        '''
        This function can be used to save the MHE results, possibly including
        solver runtimes, log files etc.
        '''

        pass


    def publish(self):

        '''
        This function can be used to publish the MHE results, e. g., via a
        message broker, for communication with other processes
        '''

        pass

