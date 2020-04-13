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

from collections import OrderedDict

class System(object):

    def _setup_system_dimensions(self):

        self.nx = 39
        self.nx_aux = 11
        self.nb = 2
        self.nu = 6
        self.nc = 5
        self.ny = 22
        self.nw = 11


    def _setup_system_components(self):

        # Model parameters

        self.p = {

            # Media

            "rho_w": 1.0e3,
            "c_w": 4.182e3,
            "rho_sl": 1.0e3,
            "c_sl": 3.8e3,
            "rho_a": 1.2,
            "c_a": 1.005e3,

            # Storages

            "V_hts": 2.0,
            "V_lts": 1.0,
            "lambda_hts": [1.15482, 2.89951, 1.195370, 1.000000],
            "eps_hts": 1e-2,

            # Building

            "C_r_c": [4.60e7, 1.4e6, 1.4e6],
            "C_r_a": 9.70e5,

            "R_r_c_amb": [0.0323477, 0.524225, 0.524225],
            "R_r_a_amb": 0.409626,

            "R_r_a_c": [0.00586922, 0.0239197, 0.0239197],
            "R_r_a_a": 0.00742401,

            "r_I_dir_c": [0.535180122166272, 0.039276989244979195, 0.0505736980484736],
            "r_I_diff_c": [0.6521733085232639, 0.0439466540793984, 0.047162806036684796],

            "r_I_dir_a": [0.0, 0.043498352810457594, 0.157981031053824],
            "r_I_diff_a": [0.0, 0.08382529976029439, 0.217300070093568],

            "A_r_c": [189.2, 189.2, 189.2],
            "A_r_w": 189.2,
            "r_r_w": 0.444832,

            "r_Qdot_fcu_a": [0.0136789, 0.0185306, 0.0],

            # Cooling system

            "C_fcu_w": 60220.8,
            "C_fcu_a": 795.97,
            "mdot_fcu_a": 1.728,
            "lambda_fcu": 1500.0,

            "mdot_lc_max": 0.5,

            # Flat plate collectors

            "eta_fpsc": 0.857461,
            "A_fpsc": 53.284,
            "C_fpsc": 248552.0,
            "alpha_fpsc": 1.97512,

            "V_fpsc_s": 2.0e-3,
            "lambda_fpsc_s": 1.0,

            # Tube collectors

            "eta_vtsc": 0.835968,
            "A_vtsc": 31.136,
            "C_vtsc": 445931.0,
            "alpha_vtsc": 1.76747,

            "V_vtsc_s": 2.0e-3,
            "lambda_vtsc_s": 1.0,

            # Pipes connecting solar collectors and solar heat exchanger

            "lambda_psc": 133.144,
            "C_psc": 378160.0,

            # Solar heat exchanger

            "V_shx": 3.8e-3,
            "A_shx": 4.02,
            "alpha_shx": 22535.7,

            "mdot_ssc_max": 0.625,

            # ACM

            "mdot_ac_ht": 0.693,
            "T_ac_ht_min": 55.0,
            "T_ac_ht_max": 95.0,

            "mdot_ac_lt": 0.8,
            "T_ac_lt_min": 10.0,
            "T_ac_lt_max": 26.0,

            "T_ac_mt_min": 15.0,
            "T_ac_mt_max": 40.0,

            # Recooling tower

            "dT_rc": 2.0,

            # Free cooling

            "mdot_fc_lt": 0.693,

            # Ambient data time constants

            "tau_T_amb" : 3600,
            "tau_I_vtsc" : 3600,
            "tau_I_fpsc" : 3600,

            # Building heat flow noise time constants

            "tau_Qdot_n_c": 3600,
            "tau_Qdot_n_a": 3600,

        }

        # States

        self.x_index = OrderedDict([

            ("T_hts", [0, 1, 2, 3]),
            ("T_lts", [4, 5]),
            ("T_fpsc", 6),
            ("T_fpsc_s", 7),
            ("T_vtsc", 8),
            ("T_vtsc_s", 9),
            ("T_pscf", 10),
            ("T_pscr", 11),
            ("T_shx_psc", [12, 13, 14, 15]),
            ("T_shx_ssc", [16, 17, 18, 19]),
            ("T_fcu_a", 20),
            ("T_fcu_w", 21),
            ("T_r_c", [22, 23, 24]),
            ("T_r_a", [25, 26, 27])

            ])

        self.x_aux_index = OrderedDict([

            ("dT_amb", 28),
            ("dI_vtsc", 29),
            ("dI_fpsc", 30),

            ("Qdot_n_c", [31, 32, 33]),
            ("Qdot_n_a", [34, 35, 36]),

            ("dalpha_vtsc", 37),
            ("dalpha_fpsc", 38)

            ])


        # Continuous controls

        self.u_index = OrderedDict([

            ("v_ppsc", 0),
            ("p_mpsc", 1),
            ("v_plc", 2),
            ("v_pssc", 3),

            ("mdot_o_hts_b", 4),
            ("mdot_i_hts_b", 5),

        ])


        # Discrete controls

        self.b_index = OrderedDict([

            ("b_ac", 0),
            ("b_fc", 1),

        ])


        # Time-varying parameters

        self.c_index = OrderedDict([

            ("T_amb", 0),
            ("I_fpsc", 1),
            ("I_vtsc", 2),
            ("I_r_dir", 3),
            ("I_r_diff", 4),

        ])


        # Process noise

        self.w_index = OrderedDict([

            ("dT_amb", 0),
            ("dI_vtsc", 1),
            ("dI_fpsc", 2),

            ("Qdot_n_c", [3, 4, 5]),
            ("Qdot_n_a", [6, 7, 8]),

            ("dalpha_vtsc", 9),
            ("dalpha_fpsc", 10)

            ])


        # Measurements

        self.y_index = OrderedDict([

            ("T_hts", [0, 1, 2, 3]),
            ("T_lts", [4, 5]),
            ("T_fpsc_s", 6),
            ("T_vtsc_s", 7),
            ("T_shx_psc", [8, 9]),
            ("T_shx_ssc", [10, 11]),
            ("T_fcu_w", 12),
            ("T_r_c", [13, 14, 15]),
            ("T_r_a", [16, 17, 18]),
            ("T_amb", 19),
            ("I_vtsc", 20),
            ("I_fpsc", 21),

            ])


    def _setup_operation_specifications_and_limits(self):

        self.p_op = {

            "T": {
                "min": 5,
                "max": 98.0
            },

            "T_sc" : {
                "T_sc_so": 65.0,
                "v_ppsc_so": 1.0,
                "T_feed_max": 85.0
            },

            "p_mpsc": {
                "min_mpc": 0.25,
                "min_real": 0.25,
                "max": 0.884, # 0.9
            },

            "v_ppsc": {
                "min_mpc": 0.0,
                "min_real": 0.3,
                "max": 1.0
            },

            "dp_mpsc": {
                "min_mpc": -0.015,
                "min_real": -0.015,
                "max": 0.015
            },            

            "v_plc": {
                "min_mpc": 0.0,
                "min_real": 0.3,
                "max": 1.0
            },

            "v_pssc": {
                "min_mpc": 0.0,
                "min_real": 0.3,
                "max": 1.0
            },

            "room": {
                "T_r_a_min": 21.5,
                "T_r_a_max": 22.5,
            },

            "acm": {
                "min_up_time": [3600.0, 900.0],
                "min_down_time": [1800.0, 900.0]
            }

        }

    
    def _setup_simulation_control_parameters(self):

        self.p_csim = {

            "dT_ac_ht": 5.0,
            "dT_ac_lt": 1.0,
            "dT_ac_mt": 1.0,

            "dT_sc_ub": 15.0,
            "dT_sc_lb": 5.0,

            "dT_vtsc_fpsc_ub": 5.0,
            "dT_vtsc_fpsc_lb": -5.0,

            "dT_fcu_w_ub": 2.0,
            "dT_fcu_w_lb": 0.0,

            "dT_o_hts_b_ub": 10.0,
            "dT_o_hts_b_lb": 0.0,

            "T_i_hts_b_active": 80.0, 
            "dT_i_hts_b_ub": 10.0,
            "dT_i_hts_b_lb": 0.0,

        }


    def __init__(self):

        self._setup_system_dimensions()
        self._setup_system_components()
        self._setup_operation_specifications_and_limits()
        self._setup_simulation_control_parameters()


    def _setup_model(self):

        # States

        self.x = ca.MX.sym("x", self.nx)

        T_hts = self.x[self.x_index["T_hts"]]
        T_lts = self.x[self.x_index["T_lts"]]

        T_fpsc = self.x[self.x_index["T_fpsc"]]
        T_fpsc_s = self.x[self.x_index["T_fpsc_s"]]
        T_vtsc = self.x[self.x_index["T_vtsc"]]
        T_vtsc_s = self.x[self.x_index["T_vtsc_s"]]

        T_pscf = self.x[self.x_index["T_pscf"]]
        T_pscr = self.x[self.x_index["T_pscr"]]

        T_shx_psc = self.x[self.x_index["T_shx_psc"]]
        T_shx_ssc = self.x[self.x_index["T_shx_ssc"]]

        T_fcu_a = self.x[self.x_index["T_fcu_a"]]
        T_fcu_w = self.x[self.x_index["T_fcu_w"]]

        T_r_c = self.x[self.x_index["T_r_c"]]
        T_r_a = self.x[self.x_index["T_r_a"]]

        dT_amb = self.x[self.x_aux_index["dT_amb"]]
        dI_vtsc = self.x[self.x_aux_index["dI_vtsc"]]
        dI_fpsc = self.x[self.x_aux_index["dI_fpsc"]]

        Qdot_n_c = self.x[self.x_aux_index["Qdot_n_c"]]
        Qdot_n_a = self.x[self.x_aux_index["Qdot_n_a"]]

        dalpha_vtsc = self.x[self.x_aux_index["dalpha_vtsc"]]
        dalpha_fpsc = self.x[self.x_aux_index["dalpha_fpsc"]]


        # Discrete controls

        self.b = ca.MX.sym("b", self.nb)

        b_ac = self.b[self.b_index["b_ac"]]
        b_fc = self.b[self.b_index["b_fc"]]


        # Continuous controls

        self.u = ca.MX.sym("u", self.nu)

        v_ppsc = self.u[self.u_index["v_ppsc"]]
        p_mpsc = self.u[self.u_index["p_mpsc"]]
        v_plc = self.u[self.u_index["v_plc"]]
        v_pssc = self.u[self.u_index["v_pssc"]]

        mdot_o_hts_b = self.u[self.u_index["mdot_o_hts_b"]]
        mdot_i_hts_b = self.u[self.u_index["mdot_i_hts_b"]]


        # Time-varying parameters

        self.c = ca.MX.sym("c", self.nc)

        T_amb = self.c[self.c_index["T_amb"]] + dT_amb
        I_vtsc = self.c[self.c_index["I_vtsc"]] + dI_vtsc
        I_fpsc = self.c[self.c_index["I_fpsc"]] + dI_fpsc
        I_r_dir = self.c[self.c_index["I_r_dir"]]
        I_r_diff = self.c[self.c_index["I_r_diff"]]


        # Process noise

        self.w = ca.MX.sym("w", self.nw)

        w_dT_amb = self.w[self.w_index["dT_amb"]]
        w_dI_vtsc = self.w[self.w_index["dI_vtsc"]]
        w_dI_fpsc = self.w[self.w_index["dI_fpsc"]]

        w_Qdot_n_c = self.w[self.w_index["Qdot_n_c"]]
        w_Qdot_n_a = self.w[self.w_index["Qdot_n_a"]]

        w_dalpha_vtsc = self.w[self.w_index["dalpha_vtsc"]]
        w_dalpha_fpsc = self.w[self.w_index["dalpha_fpsc"]]


        # Modeling

        f = []


        # Grey box ACM model

        char_curve_acm = ca.veccat(1.0, \
            T_lts[0], T_hts[0], (T_amb + self.p["dT_rc"]), \
            T_lts[0]**2, T_hts[0]**2, (T_amb + self.p["dT_rc"])**2, \
            T_lts[0] * T_hts[0], T_lts[0] * (T_amb + self.p["dT_rc"]), \
            T_hts[0] * (T_amb + self.p["dT_rc"]), \
            T_lts[0] * T_hts[0] * (T_amb + self.p["dT_rc"]))


        params_COP_ac = np.asarray([2.03268, -0.116526, -0.0165648, \
            -0.043367, -0.00074309, -0.000105659, -0.00172085, \
            0.00113422, 0.00540221, 0.00116735, -3.87996e-05])

        params_Qdot_ac_lt = np.asarray([-5.6924, 1.12102, 0.291654, \
            -0.484546, -0.00585722, -0.00140483, 0.00795341, \
            0.00399118, -0.0287113, -0.00221606, 5.42825e-05])


        COP_ac = ca.mtimes(np.atleast_2d(params_COP_ac), char_curve_acm)
        Qdot_ac_lt = ca.mtimes(np.atleast_2d(params_Qdot_ac_lt), char_curve_acm) * 1e3

        T_ac_ht = T_hts[0] - ((Qdot_ac_lt / COP_ac) / (self.p["mdot_ac_ht"] * self.p["c_w"]))
        T_ac_lt = T_lts[0] - (Qdot_ac_lt / (self.p["mdot_ac_lt"] * self.p["c_w"]))


        # Free cooling

        T_fc_lt = T_amb + self.p["dT_rc"]


        # HT storage model

        mdot_ssc = self.p["mdot_ssc_max"] * v_pssc

        m_hts = (self.p["V_hts"] * self.p["rho_w"]) / T_hts.numel()

        mdot_hts_t_s = mdot_ssc - b_ac * self.p["mdot_ac_ht"]
        mdot_hts_t_sbar = ca.sqrt(mdot_hts_t_s**2 + self.p["eps_hts"])

        mdot_hts_b_s = mdot_i_hts_b - mdot_o_hts_b
        mdot_hts_b_sbar = ca.sqrt(mdot_hts_b_s**2 + self.p["eps_hts"])

        f.append( \

            (1.0 / m_hts) * ( \

                mdot_ssc * T_shx_ssc[-1] \
                - b_ac * self.p["mdot_ac_ht"] * T_hts[0] \
                - (mdot_hts_t_s * ((T_hts[0] + T_hts[1]) / 2.0) \
                   + mdot_hts_t_sbar * ((T_hts[0] - T_hts[1]) / 2.0)) \
                - (self.p["lambda_hts"][0] / self.p["c_w"] * (T_hts[0] - T_amb))) \
            )


        f.append( \

                (1.0 / m_hts) * ( \

                    (b_ac * self.p["mdot_ac_ht"] - mdot_i_hts_b) * T_ac_ht
                    - (mdot_ssc - mdot_o_hts_b) * T_hts[1]
                    + (mdot_hts_t_s * ((T_hts[0] + T_hts[1]) / 2.0) + mdot_hts_t_sbar * ((T_hts[0] - T_hts[1]) / 2.0))
                    + (mdot_hts_b_s * ((T_hts[2] + T_hts[1]) / 2.0) + mdot_hts_b_sbar * ((T_hts[2] - T_hts[1]) / 2.0))
                    - (self.p["lambda_hts"][3] / self.p["c_w"] * (T_hts[1] - T_amb)))
            )


        f.append( \

            (1.0 / m_hts) * ( \

                mdot_hts_b_s * (((T_hts[-1] + T_hts[-2]) / 2.0) - ((T_hts[-2] + T_hts[-3]) / 2.0)) \
                   + mdot_hts_b_sbar * (((T_hts[-1] - T_hts[-2]) / 2.0) - ((T_hts[-2] - T_hts[-3]) / 2.0))
                - (self.p["lambda_hts"][-2] / self.p["c_w"] * (T_hts[-2] - T_amb)))
            )


        f.append( \

                (1.0 / m_hts) * ( \

                mdot_i_hts_b * T_ac_ht
                - mdot_o_hts_b * T_hts[-1]
                - (mdot_hts_b_s * ((T_hts[-1] + T_hts[-2]) / 2.0) + mdot_hts_b_sbar * ((T_hts[-1] - T_hts[-2]) / 2.0))
                - (self.p["lambda_hts"][-1] / self.p["c_w"] * (T_hts[-1] - T_amb)))

            )


        # LT storage model

        mdot_lc = self.p["mdot_lc_max"] * v_plc

        m_lts = (self.p["V_lts"] * self.p["rho_w"]) / T_lts.numel()

        f.append( \

            (1.0 / m_lts) * ( \
                mdot_lc * T_fcu_w \
                - (1.0 - b_ac - b_fc) * mdot_lc * T_lts[0] \
                + b_ac * (self.p["mdot_ac_lt"] - mdot_lc) * T_lts[1] \
                - b_ac * self.p["mdot_ac_lt"] * T_lts[0]
                + b_fc * (self.p["mdot_fc_lt"] - mdot_lc) * T_lts[1] \
                - b_fc * self.p["mdot_fc_lt"] * T_lts[0])
            )

        f.append( \

            (1.0 / m_lts) * ( \
                (1.0 - b_ac - b_fc) * mdot_lc * T_lts[-2] \
                - mdot_lc * T_lts[-1] \
                + b_ac * self.p["mdot_ac_lt"] * T_ac_lt \
                - b_ac * (self.p["mdot_ac_lt"] - mdot_lc) * T_lts[-1]
                + b_fc * self.p["mdot_fc_lt"] * T_fc_lt
                - b_fc * (self.p["mdot_fc_lt"] - mdot_lc) * T_lts[-1])
            )

        # Flat plate collectors

        data_v_ppsc = [-0.1, 0.0, 0.4, 0.6, 0.8, 1.0, 1.1]
        data_p_mpsc = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        data_mdot_fpsc = np.array([0.00116667, 0.        , 0.02765   , 0.03511667, 0.04258333,
                                  0.04993333, 0.04993333, 0.00116667, 0.        , 0.02765   ,
                                  0.03511667, 0.04258333, 0.04993333, 0.04993333, 0.0035    ,
                                  0.        , 0.08343   , 0.11165333, 0.13568   , 0.15895333,
                                  0.15895333, 0.005     , 0.        , 0.12508167, 0.16568   ,
                                  0.20563333, 0.2397    , 0.2397    , 0.0055    , 0.        ,
                                  0.13999333, 0.1859    , 0.22790167, 0.26488   , 0.26488   ,
                                  0.006     , 0.        , 0.14969167, 0.19844   , 0.24394167,
                                  0.28695333, 0.28695333, 0.00633333, 0.        , 0.15706667,
                                  0.21070833, 0.25807   , 0.310775  , 0.310775  , 0.0075    ,
                                  0.        , 0.17047833, 0.229775  , 0.28826667, 0.34190833,
                                  0.34190833, 0.0095    , 0.        , 0.20687333, 0.27500667,
                                  0.331775  , 0.37235333, 0.37235333, 0.013     , 0.        ,
                                  0.24111667, 0.31581   , 0.38300833, 0.44705167, 0.44705167,
                                  0.013     , 0.        , 0.24111667, 0.31581   , 0.38300833,
                                  0.44705167, 0.44705167])

        iota_mdot_fpsc = ca.interpolant('iota_mdot_fpsc', 'bspline', [data_v_ppsc, data_p_mpsc], \
            data_mdot_fpsc)

        mdot_fpsc = iota_mdot_fpsc(ca.veccat(v_ppsc, p_mpsc))

        Qdot_fpsc = self.p["eta_fpsc"] * self.p["A_fpsc"] * I_fpsc
        Qdot_fpsc_amb = (self.p["alpha_fpsc"] + dalpha_fpsc) * self.p["A_fpsc"] * (T_fpsc - T_amb)

        f.append( \
            (1.0 / self.p["C_fpsc"]) * \
            (mdot_fpsc * self.p["c_sl"] * (T_pscf - T_fpsc) + \
                Qdot_fpsc - Qdot_fpsc_amb))


        f.append((1.0 / (self.p["V_fpsc_s"] * self.p["rho_sl"] * self.p["c_sl"])) * ( \
                mdot_fpsc * self.p["c_sl"] * (T_fpsc - T_fpsc_s) - \
                self.p["lambda_fpsc_s"] * (T_fpsc_s - T_amb))
            )


        # Tube collectors

        data_mdot_vtsc = np.array([0.0155    , 0.        , 0.36735   , 0.46655   , 0.56575   ,
                                  0.6634    , 0.6634    , 0.0155    , 0.        , 0.36735   ,
                                  0.46655   , 0.56575   , 0.6634    , 0.6634    , 0.01316667,
                                  0.        , 0.32157   , 0.41501333, 0.50432   , 0.59438   ,
                                  0.59438   , 0.01166667, 0.        , 0.29325167, 0.37932   ,
                                  0.4577    , 0.54363333, 0.54363333, 0.01116667, 0.        ,
                                  0.28167333, 0.3641    , 0.44043167, 0.52345333, 0.52345333,
                                  0.01066667, 0.        , 0.271975  , 0.34822667, 0.42439167,
                                  0.50138   , 0.50138   , 0.01033333, 0.        , 0.25626667,
                                  0.33095833, 0.39859667, 0.464225  , 0.464225  , 0.00916667,
                                  0.        , 0.217855  , 0.275225  , 0.3384    , 0.39975833,
                                  0.39975833, 0.00716667, 0.        , 0.15479333, 0.19832667,
                                  0.243225  , 0.30098   , 0.30098   , 0.00366667, 0.        ,
                                  0.06721667, 0.08752333, 0.10865833, 0.13128167, 0.13128167,
                                  0.00366667, 0.        , 0.06721667, 0.08752333, 0.10865833,
                                  0.13128167, 0.13128167])

        iota_mdot_vtsc = ca.interpolant('iota_mdot_vtsc', 'bspline', [data_v_ppsc, data_p_mpsc], \
            data_mdot_vtsc)

        mdot_vtsc = iota_mdot_vtsc(ca.veccat(v_ppsc, p_mpsc))

        Qdot_vtsc = self.p["eta_vtsc"] * self.p["A_vtsc"] * I_vtsc
        Qdot_vtsc_amb = (self.p["alpha_vtsc"] + dalpha_vtsc) * self.p["A_vtsc"] * (T_vtsc - T_amb)

        f.append( \
            (1.0 / self.p["C_vtsc"]) * \
            (mdot_vtsc * self.p["c_sl"] * (T_pscf - T_vtsc) + \
                Qdot_vtsc - Qdot_vtsc_amb))

        f.append((1.0 / (self.p["V_vtsc_s"] * self.p["rho_sl"] * self.p["c_sl"])) * ( \
                mdot_vtsc * self.p["c_sl"] * (T_vtsc - T_vtsc_s) - \
                self.p["lambda_vtsc_s"] * (T_vtsc_s - T_amb))
            )


        # Pipes connecting solar collectors and solar heat exchanger

        f.append(1.0 / self.p["C_psc"] * ((mdot_fpsc + mdot_vtsc) * \
            self.p["c_sl"] * (T_shx_psc[-1] - T_pscf) - \
            self.p["lambda_psc"] * (T_pscf - T_amb)))

        f.append(1.0 / self.p["C_psc"] * (mdot_fpsc * self.p["c_sl"] * T_fpsc_s \
           + mdot_vtsc * self.p["c_sl"] * T_vtsc_s \
           - (mdot_fpsc + mdot_vtsc) * self.p["c_sl"] * T_pscr \
           - self.p["lambda_psc"] * (T_pscr - T_amb)))


        # Solar heat exchanger

        m_shx_psc = self.p["V_shx"] * self.p["rho_sl"] / T_shx_psc.numel()
        m_shx_ssc = self.p["V_shx"] * self.p["rho_w"] / T_shx_psc.numel()

        A_shx_k = self.p["A_shx"] / T_shx_psc.numel()

        f.append( \
            (1.0 / (m_shx_psc * self.p["c_sl"])) * ( \
                (mdot_fpsc + mdot_vtsc) \
                * self.p["c_sl"] * (T_pscr - T_shx_psc[0]) \
                - (A_shx_k * self.p["alpha_shx"] * (T_shx_psc[0] - T_shx_ssc[-1]))))

        for k in range(1, T_shx_psc.numel()):

            f.append( \
                (1.0 / (m_shx_psc * self.p["c_sl"])) * ( \
                    (mdot_fpsc + mdot_vtsc) \
                    * self.p["c_sl"] * (T_shx_psc[k-1] - T_shx_psc[k]) \
                    - (A_shx_k * self.p["alpha_shx"] * (T_shx_psc[k] - T_shx_ssc[-1-k]))))

        f.append( \
            (1.0 / (m_shx_ssc * self.p["c_w"])) * ( \
                (mdot_ssc - mdot_o_hts_b) * self.p["c_w"] * (T_hts[1] - T_shx_ssc[0]) \
                + mdot_o_hts_b * self.p["c_w"] * (T_hts[-1] - T_shx_ssc[0]) \
                + (A_shx_k * self.p["alpha_shx"] * (T_shx_psc[-1] - T_shx_ssc[0]))))

        for k in range(1, T_shx_ssc.numel()):

            f.append( \
                (1.0 / (m_shx_ssc * self.p["c_w"])) * ( \
                    mdot_ssc * self.p["c_w"] * (T_shx_ssc[k-1] - T_shx_ssc[k]) \
                    + (A_shx_k * self.p["alpha_shx"] * (T_shx_psc[-1-k] - T_shx_ssc[k]))))
            

        # Cooling system model

        Qdot_fcu_r = self.p["lambda_fcu"] * (T_fcu_a - T_fcu_w)

        f.append((1.0 / self.p["C_fcu_a"]) * \
            (self.p["c_a"] * self.p["mdot_fcu_a"] * (T_r_a[0] - T_fcu_a) - Qdot_fcu_r))

        f.append((1.0 / self.p["C_fcu_w"]) * \
            (self.p["c_w"] * mdot_lc * (T_lts[-1] - T_fcu_w) + Qdot_fcu_r))

        # Building model

        Qdot_r_a_c = []

        for k in range(3):

            Qdot_r_a_c.append((1.0 / self.p["R_r_a_c"][k]) * (T_r_a[k] - T_r_c[k]))

            f.append((1.0 / self.p["C_r_c"][k]) * (Qdot_r_a_c[k] \
                + (1.0 / self.p["R_r_c_amb"][k]) * (T_amb - T_r_c[k]) \
                + self.p["r_I_dir_c"][k] * I_r_dir + self.p["r_I_diff_c"][k] * I_r_diff \
                + Qdot_n_c[k]))

        Qdot_r_a_a = []
        Qdot_r_a_a.append((1.0 / self.p["R_r_a_a"]) * (T_r_a[1] - T_r_a[0]))
        Qdot_r_a_a.append((1.0 / self.p["R_r_a_a"]) * (T_r_a[2] - T_r_a[1]))

        f.append((1.0 / self.p["C_r_a"]) * (-Qdot_r_a_c[0] \
            + (1.0 / self.p["R_r_a_amb"]) * (T_amb - T_r_a[0]) \
            + Qdot_r_a_a[0] - self.p["r_Qdot_fcu_a"][0] * Qdot_fcu_r \
            + self.p["r_I_dir_a"][0] * I_r_dir + self.p["r_I_diff_a"][0] * I_r_diff \
            + Qdot_n_a[0]))

        f.append((1.0 / self.p["C_r_a"]) * (-Qdot_r_a_c[1] \
            + (1.0 / self.p["R_r_a_amb"]) * (T_amb - T_r_a[1]) \
            - Qdot_r_a_a[0] + Qdot_r_a_a[1] - self.p["r_Qdot_fcu_a"][1] * Qdot_fcu_r \
            + self.p["r_I_dir_a"][1] * I_r_dir + self.p["r_I_diff_a"][1] * I_r_diff \
            + Qdot_n_a[1]))

        f.append((1.0 / self.p["C_r_a"]) * (-Qdot_r_a_c[2] \
            + (1.0 / self.p["R_r_a_amb"]) * (T_amb - T_r_a[2]) \
            - Qdot_r_a_a[1] - self.p["r_Qdot_fcu_a"][2] * Qdot_fcu_r \
            + self.p["r_I_dir_a"][2] * I_r_dir + self.p["r_I_diff_a"][2] * I_r_diff \
            + Qdot_n_a[2]))

        # Ambient data deviations

        f.append((-dT_amb / self.p["tau_T_amb"]) + w_dT_amb)
        f.append((-dI_vtsc / self.p["tau_I_vtsc"]) + w_dI_vtsc)
        f.append((-dI_fpsc / self.p["tau_I_fpsc"]) + w_dI_fpsc)

        # Heat flow noise on building model

        for k in range(3):
            f.append((-Qdot_n_c[k] / self.p["tau_Qdot_n_c"]) + w_Qdot_n_c[k])

        for k in range(3):
            f.append((-Qdot_n_a[k] / self.p["tau_Qdot_n_a"]) + w_Qdot_n_a[k])

        # Heat loss noise for solar collectors

        f.append(w_dalpha_vtsc)
        f.append(w_dalpha_fpsc)


        self.f = ca.veccat(*f)


        # Measurement function

        self.y = ca.MX.sym("y", self.ny)

        h = []

        h.append(T_hts[0])
        h.append(T_hts[1])
        h.append(T_hts[2])
        h.append(T_hts[3])

        h.append(T_lts[0])
        h.append(T_lts[1])

        h.append(T_fpsc_s)
        h.append(T_vtsc_s)

        h.append(T_shx_psc[0])
        h.append(T_shx_psc[3])

        h.append(T_shx_ssc[0])
        h.append(T_shx_ssc[3])

        h.append(T_fcu_w)

        h.append(T_r_c[0])
        h.append(T_r_c[1])
        h.append(T_r_c[2])

        h.append(T_r_a[0])
        h.append(T_r_a[1])
        h.append(T_r_a[2])

        h.append(T_amb)
        h.append(I_vtsc)
        h.append(I_fpsc)

        self.h = ca.veccat(*h)


    def _remove_unserializable_attributes(self):

        delattr(self, "x")
        delattr(self, "u")
        delattr(self, "b")
        delattr(self, "c")
        delattr(self, "w")
        delattr(self, "f")
        delattr(self, "y")
        delattr(self, "h")

