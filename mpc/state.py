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

import logging
import numpy as np
import pandas as pd

from system import System

logger = logging.getLogger(__name__)

class State(System):

    @property
    def x_hat(self):

        x_index = []

        for index in [self.x_index, self.x_aux_index]:
        
            for x in index:

                try:

                    for j, idx in enumerate(index[x]):

                        x_index.append(x.lower() + "_" + str(j))

                except TypeError:

                    x_index.append(x.lower())

        try:

            return np.squeeze(self._df_state[x_index].values)

        except AttributeError:

            msg = "System state not available yet, call update() first."
            
            logging.error(msg)
            raise RuntimeError(msg)


    def _get_current_system_state(self):

        '''
        Generate some generic state data for demonstrational purposes.
        '''

        data = {'dalpha_fpsc': {'2020-04-10 08:00:00.0+0000': 0.0},
            'dalpha_vtsc': {'2020-04-10 08:00:00.0+0000': 0.0},
            'di_fpsc': {'2020-04-10 08:00:00.0+0000': 0.0},
            'di_vtsc': {'2020-04-10 08:00:00.0+0000': 0.0},
            'dt_amb': {'2020-04-10 08:00:00.0+0000': 0.0},
            'qdot_n_a_0': {'2020-04-10 08:00:00.0+0000': 0.0},
            'qdot_n_a_1': {'2020-04-10 08:00:00.0+0000': 0.0},
            'qdot_n_a_2': {'2020-04-10 08:00:00.0+0000': 0.0},
            'qdot_n_c_0': {'2020-04-10 08:00:00.0+0000': 0.0},
            'qdot_n_c_1': {'2020-04-10 08:00:00.0+0000': 0.0},
            'qdot_n_c_2': {'2020-04-10 08:00:00.0+0000': 0.0},
            't_fcu_a': {'2020-04-10 08:00:00.0+0000': 22.0},
            't_fcu_w': {'2020-04-10 08:00:00.0+0000': 25.0},
            't_hts_0': {'2020-04-10 08:00:00.0+0000': 52.0},
            't_hts_1': {'2020-04-10 08:00:00.0+0000': 50.0},
            't_hts_2': {'2020-04-10 08:00:00.0+0000': 48.0},
            't_hts_3': {'2020-04-10 08:00:00.0+0000': 45.0},
            't_shx_ssc_0': {'2020-04-10 08:00:00.0+0000': 32.0},
            't_shx_ssc_1': {'2020-04-10 08:00:00.0+0000': 32.0},
            't_shx_ssc_2': {'2020-04-10 08:00:00.0+0000': 32.0},
            't_shx_ssc_3': {'2020-04-10 08:00:00.0+0000': 32.0},
            't_shx_psc_0': {'2020-04-10 08:00:00.0+0000': 11.0},
            't_shx_psc_1': {'2020-04-10 08:00:00.0+0000': 11.0},
            't_shx_psc_2': {'2020-04-10 08:00:00.0+0000': 11.0},
            't_shx_psc_3': {'2020-04-10 08:00:00.0+0000': 11.0},
            't_lts_0': {'2020-04-10 08:00:00.0+0000': 20.0},
            't_lts_1': {'2020-04-10 08:00:00.0+0000': 19.0},
            't_pscf': {'2020-04-10 08:00:00.0+0000': 18.0},
            't_pscr': {'2020-04-10 08:00:00.0+0000': 20.0},
            't_r_a_0': {'2020-04-10 08:00:00.0+0000': 24.0},
            't_r_a_1': {'2020-04-10 08:00:00.0+0000': 23.0},
            't_r_a_2': {'2020-04-10 08:00:00.0+0000': 23.0},
            't_r_c_0': {'2020-04-10 08:00:00.0+0000': 24.0},
            't_r_c_1': {'2020-04-10 08:00:00.0+0000': 24.0},
            't_r_c_2': {'2020-04-10 08:00:00.0+0000': 23.5},
            't_fpsc': {'2020-04-10 08:00:00.0+0000': 20.0},
            't_fpsc_s': {'2020-04-10 08:00:00.0+0000': 20.0},
            't_vtsc': {'2020-04-10 08:00:00.0+0000': 22.0},
            't_vtsc_s': {'2020-04-10 08:00:00.0+0000': 22.0}}

        df_state = pd.DataFrame(data)
        df_state.index = pd.to_datetime(df_state.index)

        self._df_state = df_state


    def update(self):

        self._get_current_system_state()


if __name__ == "__main__":

    state = State()
    state.update()

    print(state.x_hat)

