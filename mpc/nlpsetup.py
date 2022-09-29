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
import numpy as np
import casadi as ca
import subprocess

from system import System

import logging

logger = logging.getLogger(__name__)

class NLPSetupBaseClass(System):

    _PATH_TO_NLP_SOURCE = "src/"
    _PATH_TO_NLP_OBJECT = "lib/"
    
    _CXX_COMPILERS = ["clang", "gcc"]
    _CXX_FLAGS = ["-fPIC", "-v", "-shared", "-fno-omit-frame-pointer"]
    _CXX_FLAG_NO_OPT = ["-O0"]
    _CXX_FLAG_OPT = ["-O1"]


    def _setup_timing(self, timing):

        self._timing = timing


    def _setup_collocation_options(self):

        self.d = 3
        self.tau_root = [0] + ca.collocation_points(self.d, 'radau')


    def _setup_directories(self):

        if not os.path.exists(self._PATH_TO_NLP_OBJECT):
            os.mkdir(self._PATH_TO_NLP_OBJECT)
        
        if not os.path.exists(self._PATH_TO_NLP_SOURCE):
            os.mkdir(self._PATH_TO_NLP_SOURCE)


    def _export_nlp_to_c_code(self):

        __dirname__ = os.path.dirname(os.path.abspath(__file__))

        nlpsolver = ca.nlpsol("nlpsol", "ipopt", self.nlp)
        nlpsolver.generate_dependencies(self._NLP_SOURCE_FILENAME)

        os.rename(self._NLP_SOURCE_FILENAME, os.path.join(__dirname__, 
            self._PATH_TO_NLP_SOURCE, self._NLP_SOURCE_FILENAME))


    def _check_compiler_availability(self):

        logging.info("Checking if a C compiler is available ...")

        for compiler in self._CXX_COMPILERS:

            try:
                return_status = subprocess.call([compiler, "--version"], shell=False)
                if return_status == 0:
                    logging.info("Compiler " + compiler + " found.")

                    self._compiler = compiler
                    return

            except FileNotFoundError:
                logging.info("Compiler " + compiler + " not found, trying other compiler ...")

        raise RuntimeError("No C compiler found, NLP object cannot be generated.")


    def _compile_nlp_object(self, optimize_for_speed, overwrite_existing_object):

        '''
        When optimize_for_speed=True, the NLP objects are compiled to facilitate
        faster computation; however, more time and memory is required for compiltation
        '''

        __dirname__ = os.path.dirname(os.path.abspath(__file__))

        path_to_nlp_source_code = os.path.join(__dirname__, self._PATH_TO_NLP_SOURCE, \
            self._NLP_SOURCE_FILENAME)
        path_to_nlp_object = os.path.join(__dirname__, self._PATH_TO_NLP_OBJECT, \
            self._NLP_OBJECT_FILENAME)

        if not os.path.isfile(path_to_nlp_object) or overwrite_existing_object:

            logger.info("Compiling NLP object ...")

            compiler_flags = self._CXX_FLAGS

            if optimize_for_speed:
                compiler_flags += self._CXX_FLAG_OPT
            else:
                compiler_flags += self._CXX_FLAG_NO_OPT

            compiler_command = [self._compiler] + compiler_flags + \
                ["-o"] + [path_to_nlp_object] + [path_to_nlp_source_code]

            return_status = subprocess.call(compiler_command, shell=False)

            if return_status == 0:

                logger.info("Compilation of NLP object finished successfully.")

            else:

                logger.error("A problem occurred compiling the NLP object, check compiler output for further information.")
                raise subprocess.CalledProcessError(return_status, ' '.join(compiler_command))

        else:

            logger.info("NLP object already exists, not overwriting it.")


    def generate_nlp_object(self, optimize_for_speed=False, \
        overwrite_existing_object=False):

        logger.info("Generating NLP object ...")

        self._setup_model()
        self._setup_collocation_options()
        self._setup_nlp_functions()
        self._setup_nlp()
        self._setup_directories()
        self._export_nlp_to_c_code()
        self._check_compiler_availability()
        self._compile_nlp_object(optimize_for_speed=optimize_for_speed, \
            overwrite_existing_object=overwrite_existing_object)

        logger.info("Finished generating NLP object.")


class NLPSetupMPC(NLPSetupBaseClass):

    _NLP_SOURCE_FILENAME = "nlp_mpc.c"
    _NLP_OBJECT_FILENAME = "nlp_mpc.so"


    def __init__(self, timing):

        super().__init__()

        self._setup_timing(timing)


    def _setup_nlp_functions(self):

        f = ca.Function('f', [self.x,self.c,self.u,self.b,self.w], [self.f])

        C = np.zeros((self.d+1,self.d+1))
        D = np.zeros(self.d+1)

        for j in range(self.d+1):

            p = np.poly1d([1])
            for r in range(self.d+1):
                if r != j:
                    p *= np.poly1d([1, -self.tau_root[r]]) / (self.tau_root[j]-self.tau_root[r])

            D[j] = p(1.0)

            pder = np.polyder(p)
            for r in range(self.d+1):
                C[j,r] = pder(self.tau_root[r])


        # Collocation equations

        x_k_c = [ca.MX.sym("x_k_c_"+ str(j), self.nx) for j in range(self.d+1)]
        x_k_next_c = ca.MX.sym("x_k_next_c", self.nx)
        c_k_c = ca.MX.sym("c_k_c", self.nc)
        u_k_c = ca.MX.sym("u_k_c", self.nu)
        b_k_c = ca.MX.sym("b_k_c", self.nb)
        dt_k_c = ca.MX.sym("dt_k_c")
        w_k_c = ca.MX.sym("w_k_c", self.nw)

        eq_c = []

        for j in range(1,self.d+1):
            
            x_p_c = 0

            for r in range(self.d+1):

                x_p_c += C[r,j] * x_k_c[r]

            f_k_c = f(x_k_c[j], c_k_c, u_k_c, b_k_c, w_k_c)

            eq_c.append(dt_k_c*f_k_c - x_p_c)
        
        eq_c = ca.veccat(*eq_c)
        
        xf_c = 0

        for r in range(self.d+1):
        
            xf_c += D[r] * x_k_c[r]

            eq_d = xf_c - x_k_next_c

        self.F = ca.Function("F", \
            x_k_c + [x_k_next_c, c_k_c, u_k_c, b_k_c, dt_k_c, w_k_c],
            [eq_c, eq_d], \
            ["x_k_"+ str(j) for j in range(self.d+1)] + ["x_k_next", "c_k", "u_k", "b_k", "dt_k", "w_k"], \
            ["eq_c", "eq_d"])


        # ACM operation condition equations

        s_ac_lb = ca.MX.sym("s_ac_lb", 3)

        T_ac_min = self.b[self.b_index["b_ac"]] * ca.veccat( \
            self.x[self.x_index["T_lts"][0]] - self.p["T_ac_lt_min"] + s_ac_lb[0], \
            self.x[self.x_index["T_hts"][0]] - self.p["T_ac_ht_min"] + s_ac_lb[1], \
            self.c[self.c_index["T_amb"]] + self.p["dT_rc"] - self.p["T_ac_mt_min"] + s_ac_lb[2])

        self.T_ac_min_fcn = ca.Function("T_ac_min_fcn", \
            [self.x, self.c, self.b, s_ac_lb], [T_ac_min])


        s_ac_ub = ca.MX.sym("s_ac_ub", 3)

        T_ac_max = self.b[self.b_index["b_ac"]] * ca.veccat( \
            self.x[self.x_index["T_lts"][0]] - self.p["T_ac_lt_max"] - s_ac_ub[0], \
            self.x[self.x_index["T_hts"][0]] - self.p["T_ac_ht_max"] - s_ac_ub[1], \
            self.c[self.c_index["T_amb"]] + self.p["dT_rc"] - self.p["T_ac_mt_max"] - s_ac_ub[2])

        self.T_ac_max_fcn = ca.Function("T_ac_max_fcn", \
            [self.x, self.c, self.b, s_ac_ub], [T_ac_max])


        # Objective temperature comfort range

        dT_r_a = ca.MX.sym("dT_r_a")

        self.dT_r_a_fcn = ca.Function("dT_r_a_fcn", \
            [self.x, dT_r_a], [self.x[self.x_index["T_r_a"][0]] + dT_r_a])


        # States limits soft constraints

        s_x = ca.MX.sym("s_x", self.nx-self.nx_aux)

        self.s_x_fcn = ca.Function("s_x_fcn", \
            [self.x, s_x], [self.x[:self.nx-self.nx_aux] + s_x])


        # Assure ppsc is running at high speed when collector temperature is high

        s_ppsc_fpsc = ca.MX.sym("s_ppsc_fpsc")

        self.v_ppsc_so_fpsc_fcn_1 = ca.Function("v_ppsc_so_fpsc_fcn_1", \
            [self.x, s_ppsc_fpsc], \
            [(self.x[self.x_index["T_fpsc"]] - self.p_op["T_sc"]["T_sc_so"]) \
                * (self.p_op["T_sc"]["v_ppsc_so"] - s_ppsc_fpsc)])

        self.v_ppsc_so_fpsc_fcn_2 = ca.Function("v_ppsc_so_fpsc_fcn_2", \
            [self.u, s_ppsc_fpsc], [s_ppsc_fpsc - self.u[self.u_index["v_ppsc"]]])

        s_ppsc_vtsc = ca.MX.sym("s_ppsc_vtsc")

        self.v_ppsc_so_vtsc_fcn_1 = ca.Function("v_ppsc_so_vtsc_fcn_1", \
            [self.x, s_ppsc_vtsc], \
            [(self.x[self.x_index["T_vtsc"]] - self.p_op["T_sc"]["T_sc_so"]) \
                * (self.p_op["T_sc"]["v_ppsc_so"] - s_ppsc_vtsc)])

        self.v_ppsc_so_vtsc_fcn_2 = ca.Function("v_ppsc_so_vtsc_fcn_2", \
            [self.u, s_ppsc_vtsc], [s_ppsc_vtsc - self.u[self.u_index["v_ppsc"]]])


        # Assure HTS bottom layer mass flows are always smaller or equal to
        # the corresponding total pump flow

        mdot_hts_b_max = ca.veccat( \

            self.u[self.u_index["mdot_o_hts_b"]] - self.p["mdot_ssc_max"] * self.u[self.u_index["v_pssc"]], \
            self.u[self.u_index["mdot_i_hts_b"]] - self.b[self.b_index["b_ac"]] * self.p["mdot_ac_ht"])

        self.mdot_hts_b_max_fcn = ca.Function("mdot_hts_b_max_fcn", \
            [self.u, self.b], [mdot_hts_b_max])


        # Lagrange term functional

        u_prev = ca.MX.sym("u_prev", self.nu)

        obj = ca.veccat(4*s_ac_lb**2, 1.5*s_ac_ub**2, 2*s_x**2, 1e1*dT_r_a**2, \
            0.01*self.u[self.u_index["v_ppsc"]], \
            0.001*self.u[self.u_index["v_pssc"]], \
            0.01*self.u[self.u_index["v_plc"]], \
            0.1*(u_prev[self.u_index["p_mpsc"]] - self.u[self.u_index["p_mpsc"]])**2, \
            0.1*(u_prev[self.u_index["mdot_o_hts_b"]] - self.u[self.u_index["mdot_o_hts_b"]])**2, \
            0.1*(u_prev[self.u_index["mdot_i_hts_b"]] - self.u[self.u_index["mdot_i_hts_b"]])**2)

        self.obj_fcn = ca.Function("obj_fcn", [s_ac_lb, s_ac_ub, s_x, \
            dT_r_a, self.u, u_prev, self.b], [obj])


    def _setup_nlp(self):

        # Optimization variables

        V = []

        # Constraints

        g = []

        # Lagrange term of the objective

        L = []

        # Parametric controls

        P = []

        X0 = ca.MX.sym("x_0_0", self.nx)

        V.append(X0)
        x_k_0 = X0

        u_k_prev = None

        for k in range(self._timing.N):

            # Add new states

            x_k_j = [ca.MX.sym("x_" + str(k) + "_" + str(j), self.nx) for j in range(1,self.d+1)]

            x_k  = [x_k_0] + x_k_j
            x_k_next_0 = ca.MX.sym("x_" + str(k+1) + "_0", self.nx)

            # Add new binary controls

            b_k = ca.MX.sym("b_" + str(k), self.nb)

            # Add new continuous controls

            u_k = ca.MX.sym("u_" + str(k), self.nu)

            if u_k_prev is None:

                u_k_prev = u_k

            # Add new parametric controls

            c_k = ca.MX.sym("c_" + str(k), self.nc)

            # Add new objective residual

            r_k = ca.MX.sym("r_" + str(k))

            # Add parameter for time step at current point

            dt_k = ca.MX.sym("dt_" + str(k))

            # Add parameter for process noise

            w_k = ca.MX.sym("w_" + str(k), self.nw)

            # Add collocation equations

            F_k_inp = {"x_k_" + str(i): x_k_i for i, x_k_i in enumerate(x_k)}
            F_k_inp.update({"x_k_next": x_k_next_0, "c_k": c_k, "u_k": u_k, \
                "b_k": b_k, "dt_k": dt_k, "w_k": w_k})

            F_k = self.F(**F_k_inp)

            g.append(F_k["eq_c"])
            g.append(F_k["eq_d"])

            # Add new slack variable for T_ac_min condition

            s_ac_lb_k = ca.MX.sym("s_ac_lb_" + str(k), 3)

            # Setup T_ac_min conditions

            g.append(self.T_ac_min_fcn( \
                x_k_0, c_k, b_k, s_ac_lb_k))


            # Add new slack variable for T_ac_max condition

            s_ac_ub_k = ca.MX.sym("s_ac_ub_" + str(k), 3)

            # Setup T_ac_max conditions

            g.append(self.T_ac_max_fcn( \
                x_k_0, c_k, b_k, s_ac_ub_k))

            # Setup objective temperature range condition

            g.append(self.dT_r_a_fcn(x_k_0, r_k))

            # Add new slack variable for state limits soft constraints

            s_x_k = ca.MX.sym("s_x_" + str(k), self.nx-self.nx_aux)

            # Setup state limits as soft constraints to prevent infeasibility

            g.append(self.s_x_fcn(x_k_next_0, s_x_k))

            # Assure ppsc is running at high speed when collector temperature is high

            s_ppsc_fpsc_k = ca.MX.sym("s_ppsc_fpsc_" + str(k))

            g.append(self.v_ppsc_so_fpsc_fcn_1(x_k_0, s_ppsc_fpsc_k))
            g.append(self.v_ppsc_so_fpsc_fcn_2(u_k, s_ppsc_fpsc_k))

            s_ppsc_vtsc_k = ca.MX.sym("s_ppsc_vtsc_" + str(k))

            g.append(self.v_ppsc_so_vtsc_fcn_1(x_k_0, s_ppsc_vtsc_k))
            g.append(self.v_ppsc_so_vtsc_fcn_2(u_k, s_ppsc_vtsc_k))

            # Assure HTS bottom layer mass flows are always smaller or equal to
            # the corresponding total pump flow

            g.append(self.mdot_hts_b_max_fcn(u_k, b_k))

            # Append new optimization variables, boundaries and initials

            for x_j in x_k_j:

                V.append(x_j)

            V.append(b_k)

            # SOS1 constraint

            g.append(ca.sum1(b_k))

            V.append(s_ac_lb_k)

            V.append(s_ac_ub_k)

            V.append(s_x_k)

            V.append(s_ppsc_fpsc_k)

            V.append(s_ppsc_vtsc_k)

            V.append(u_k)

            V.append(r_k)

            V.append(x_k_next_0)

            L.append(self.obj_fcn(s_ac_lb_k, s_ac_ub_k, s_x_k, \
                r_k, u_k, u_k_prev, b_k))

            P.append(c_k)
            P.append(dt_k)
            P.append(w_k)

            x_k_0 = V[-1]
            u_prev = u_k


        # Concatenate objects

        self.V = ca.veccat(*V)
        self.g = ca.veccat(*g)
        self.L = ca.veccat(*L)
        self.P = ca.veccat(*P)

        # Setup objective

        self.f = ca.sum1(self.L)

        self.nlp = {'x':self.V, 'p': self.P, 'f':self.f, 'g':self.g}


class NLPSetupMHE(NLPSetupBaseClass):

    _NLP_SOURCE_FILENAME = "nlp_mhe.c"
    _NLP_OBJECT_FILENAME = "nlp_mhe.so"


    def _setup_measurement_error_covariance_and_weightings(self):

        self.R_r = 3.0 * ca.DM.eye(self.ny)
        self.W_r = ca.inv(self.R_r)


    def _setup_process_noise_covariance_and_weightings(self):

        r_w = np.ones(self.nx)

        r_w[self.x_index["T_hts"][0]] = 2.0
        r_w[self.x_index["T_hts"][1]] = 2.0
        r_w[self.x_index["T_hts"][2]] = 2.0
        r_w[self.x_index["T_hts"][3]] = 2.0

        r_w[self.x_index["T_lts"][0]] = 2.0
        r_w[self.x_index["T_lts"][1]] = 2.0

        r_w[self.x_index["T_fpsc"]] = 1.0
        r_w[self.x_index["T_fpsc_s"]] = 2.0

        r_w[self.x_index["T_vtsc"]] = 1.0
        r_w[self.x_index["T_vtsc_s"]] = 2.0

        r_w[self.x_index["T_pscf"]] = 1.0
        r_w[self.x_index["T_pscr"]] = 1.0

        r_w[self.x_index["T_shx_psc"][0]] = 3.0
        r_w[self.x_index["T_shx_psc"][1]] = 1.0
        r_w[self.x_index["T_shx_psc"][2]] = 1.0
        r_w[self.x_index["T_shx_psc"][3]] = 3.0

        r_w[self.x_index["T_shx_ssc"][0]] = 3.0
        r_w[self.x_index["T_shx_ssc"][1]] = 1.0
        r_w[self.x_index["T_shx_ssc"][2]] = 1.0
        r_w[self.x_index["T_shx_ssc"][3]] = 3.0

        r_w[self.x_index["T_fcu_a"]] = 3.0
        r_w[self.x_index["T_fcu_w"]] = 2.0

        r_w[self.x_index["T_r_c"][0]] = 2.0
        r_w[self.x_index["T_r_c"][1]] = 2.0
        r_w[self.x_index["T_r_c"][2]] = 2.0

        r_w[self.x_index["T_r_a"][0]] = 2.0
        r_w[self.x_index["T_r_a"][1]] = 2.0
        r_w[self.x_index["T_r_a"][2]] = 2.0

        r_w[self.x_aux_index["dT_amb"]] = 2.0
        r_w[self.x_aux_index["dI_vtsc"]] = 10.0
        r_w[self.x_aux_index["dI_fpsc"]] = 10.0

        for x_aux_idx in self.x_aux_index["Qdot_n_c"]:

            r_w[x_aux_idx]  = 5.0

        for x_aux_idx in self.x_aux_index["Qdot_n_a"]:

            r_w[x_aux_idx]  = 5.0

        r_w[self.x_aux_index["dalpha_vtsc"]] = 1.0
        r_w[self.x_aux_index["dalpha_fpsc"]] = 1.0

        self.R_w = ca.diag(r_w)
        self.W_w = ca.inv(self.R_w)


    def __init__(self, timing):

        super().__init__()

        self._setup_timing(timing)
        self._setup_measurement_error_covariance_and_weightings()
        self._setup_process_noise_covariance_and_weightings()


    def _setup_nlp_functions(self):

        f = ca.Function('f', [self.x,self.c,self.u,self.b,self.w], [self.f])

        C = np.zeros((self.d+1,self.d+1))
        D = np.zeros(self.d+1)

        for j in range(self.d+1):

            p = np.poly1d([1])
            for r in range(self.d+1):
                if r != j:
                    p *= np.poly1d([1, -self.tau_root[r]]) / (self.tau_root[j]-self.tau_root[r])

            D[j] = p(1.0)

            pder = np.polyder(p)
            for r in range(self.d+1):
                C[j,r] = pder(self.tau_root[r])


        # Collocation equations

        x_k_c = [ca.MX.sym("x_k_c_"+ str(j), self.nx) for j in range(self.d+1)]
        x_k_next_c = ca.MX.sym("x_k_next_c", self.nx)
        c_k_c = ca.MX.sym("c_k_c", self.nc)
        u_k_c = ca.MX.sym("u_k_c", self.nu)
        b_k_c = ca.MX.sym("b_k_c", self.nb)
        dt_k_c = ca.MX.sym("dt_k_c")
        w_k_c = ca.MX.sym("w_k_c", self.nw)

        eq_c = []

        for j in range(1,self.d+1):

            x_p_c = 0

            for r in range(self.d+1):

                x_p_c += C[r,j] * x_k_c[r]

            f_k_c = f(x_k_c[j], c_k_c, u_k_c, b_k_c, w_k_c)

            eq_c.append(dt_k_c*f_k_c - x_p_c)

        eq_c = ca.veccat(*eq_c)

        xf_c = 0

        for r in range(self.d+1):

            xf_c += D[r] * x_k_c[r]

            eq_d = xf_c - x_k_next_c

        self.F = ca.Function("F", \
            x_k_c + [x_k_next_c, c_k_c, u_k_c, b_k_c, dt_k_c, w_k_c],
            [eq_c, eq_d], \
            ["x_k_"+ str(j) for j in range(self.d+1)] + \
                ["x_k_next", "c_k", "u_k", "b_k", "dt_k", "w_k"], \
            ["eq_c", "eq_d"])


        # Residuals function

        y_k_c = ca.MX.sym("y_k_c", self.ny)

        self.res_fcn = ca.Function("res_fcn", [self.x, y_k_c, self.c], \
            [ca.mtimes([(self.h - y_k_c).T, self.W_r, (self.h - y_k_c)])])


    def _setup_nlp(self):

        # Optimization variables

        V = []

        # Constraints

        g = []

        # Residuals for objective

        R = 0

        # Parametric controls

        P = []

        X0 = ca.MX.sym("X0", self.nx)
        V.append(X0)

        x_arr = ca.MX.sym("x_arr", self.nx)
        W_x_arr = ca.MX.sym("W_x_arr", self.nx, self.nx)

        R += ca.mtimes([(x_arr - X0).T, W_x_arr, (x_arr - X0)])

        P.append(x_arr)
        P.append(W_x_arr)

        x_k_0 = X0

        for k in range(self._timing.N):

            # Add new states

            x_k_j = [ca.MX.sym("x_" + str(k) + "_" + str(j), self.nx) for j in range(1,self.d+1)]

            x_k  = [x_k_0] + x_k_j
            x_k_next_0 = ca.MX.sym("x_" + str(k+1) + "_0", self.nx)

            # Add new binary controls

            b_k = ca.MX.sym("b_" + str(k), self.nb)

            # Add new continuous controls

            u_k = ca.MX.sym("u_" + str(k), self.nu)

            # Add new parametric controls

            c_k = ca.MX.sym("c_" + str(k), self.nc)

            # Add new measurement

            y_k = ca.MX.sym("y_" + str(k), self.ny)

            # Add parameter for time step at current point

            dt_k = ca.MX.sym("dt_" + str(k))

            # Add parameter for process noise

            w_k = ca.MX.sym("w_" + str(k), self.nw)

            # Add collocation equations

            F_k_inp = {"x_k_" + str(i): x_k_i for i, x_k_i in enumerate(x_k)}
            F_k_inp.update({"x_k_next": x_k_next_0, "c_k": c_k, "u_k": u_k, \
                "b_k": b_k, "dt_k": dt_k, "w_k": w_k})

            F_k = self.F(**F_k_inp)

            g.append(F_k["eq_c"])
            g.append(F_k["eq_d"])

            # Append measurement residuals

            R += self.res_fcn(x_k_0, y_k, c_k)

            # Append process noise

            R += ca.mtimes([w_k.T, self.W_w[-self.nw:, -self.nw:], w_k])

            # Append new optimization variables, boundaries and initials

            for x_j in x_k_j:

                V.append(x_j)

            P.append(b_k)
            P.append(u_k)
            P.append(y_k)
            P.append(c_k)
            P.append(dt_k)

            V.append(w_k)
            V.append(x_k_next_0)

            x_k_0 = V[-1]


        y_k = ca.MX.sym("y_" + str(k+1), self.ny)

        R += self.res_fcn(x_k_0, y_k, c_k)

        P.append(y_k)

        # Concatenate objects

        self.V = ca.veccat(*V)
        self.g = ca.veccat(*g)
        self.P = ca.veccat(*P)

        # Setup objective

        self.f = R

        self.nlp = {'x': self.V, 'p': self.P, 'f': self.f, 'g': self.g}


if __name__ == "__main__":

    import time
    from timing import TimingMPC, TimingMHE

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler()

    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    timing_mpc = TimingMPC(startup_time = time.time())
    nlpsetup_mpc = NLPSetupMPC(timing = timing_mpc)
    nlpsetup_mpc.generate_nlp_object(optimize_for_speed=False, \
        overwrite_existing_object=False)

    timing_mhe = TimingMHE(startup_time = time.time())
    nlpsetup_mhe = NLPSetupMHE(timing = timing_mhe)
    nlpsetup_mhe.generate_nlp_object(optimize_for_speed=False, \
        overwrite_existing_object=False)

