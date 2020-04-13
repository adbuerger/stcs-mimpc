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
import pandas as pd 
import datetime as dt

from system import System

from abc import ABCMeta, abstractmethod

import logging

logger = logging.getLogger(__name__)

class Ambient(System):

    @property
    def time_grid(self):

        return self._timing.time_grid


    @property
    def c_data(self):

        try:

            return self._df_ambient[list(self.c_index.keys())].values

        except AttributeError:

            msg = "Ambient conditions (parametric inputs) not available yet, call update() first."
            
            logging.error(msg)
            raise RuntimeError(msg)


    def _setup_timing(self, timing):

        self._timing = timing


    def __init__(self, timing):

        super().__init__()

        self._setup_timing(timing = timing)


    def _generate_generic_ambient_data(self):

        '''
        Generate some generic ambient data for demonstrational purposes.
        '''

        t0 = self._timing.time_points[0].replace(hour=0, minute=0, \
            second=0, microsecond=0) - dt.timedelta(days=1)

        tf = self._timing.time_points[0].replace(hour=23, minute=59, \
            second=59, microsecond=0) + dt.timedelta(days=1)

        time_points = pd.date_range(t0, tf, freq="5min")

        self._df_ambient = pd.DataFrame(0, index=time_points, columns=["I_fpsc"])

        self._df_ambient["I_fpsc"] += 800 * np.sin(2*np.pi * \
            (self._df_ambient["I_fpsc"].index - \
                self._df_ambient["I_fpsc"].index[0]).total_seconds() / 86400.0 - np.pi/2)
        self._df_ambient["I_fpsc"][self._df_ambient["I_fpsc"] <= 0.0] = 0.0

        self._df_ambient["I_vtsc"] = 0.8 * self._df_ambient["I_fpsc"]
        self._df_ambient["I_r_dir"] = 0.5 * self._df_ambient["I_fpsc"]
        self._df_ambient["I_r_diff"] = 0.3 * self._df_ambient["I_fpsc"]

        self._df_ambient["T_amb"] = 15
        self._df_ambient["T_amb"] += np.cos(2*np.pi * \
            (self._df_ambient["T_amb"].index - \
                self._df_ambient["T_amb"].index[0]).total_seconds() / 86400.0 + 2.3)

    def _set_ambient_time_range(self):

        self.timestamp_ambient_start = self._timing.time_points[0]
        self.timestamp_ambient_end = self.timestamp_ambient_start + dt.timedelta(seconds = self._timing.t_f)


    def _adjust_ambient_to_mpc_grid(self):

        self._df_ambient.index = self._df_ambient.index.tz_convert( \
            self._timing.timezone_name)

        self._df_ambient = self._df_ambient.reindex( \
            self._df_ambient.index.union(pd.date_range( \
                start = self.timestamp_ambient_start, \
                end = self.timestamp_ambient_end, \
                freq = str(self._timing.dt_day) + 's')))

        self._df_ambient.interpolate(method = "linear", inplace = True)

        self._df_ambient = self._df_ambient.reindex(pd.date_range( \
                start = self.timestamp_ambient_start, \
                end = self.timestamp_ambient_end, \
                freq = str(self._timing.dt_day) + 's'))


    def _convert_time_points_to_time_grid(self):

        self._df_ambient.index = np.round((self._df_ambient.index - \
            self._df_ambient.index[0]).total_seconds())

        self._df_ambient = self._df_ambient.reindex(self._timing.time_grid)


    def _fill_remaining_nan_values(self):

        self._df_ambient.interpolate(method = "linear", inplace = True)
        self._df_ambient.bfill(inplace = True)


    def update(self):

        self._generate_generic_ambient_data()

        self._set_ambient_time_range()
        self._adjust_ambient_to_mpc_grid()
        self._convert_time_points_to_time_grid()

        self._fill_remaining_nan_values()


if __name__ == "__main__":

    import time
    import datetime
    import matplotlib.pyplot as plt

    from timing import TimingMPC, TimingMHE

    startup_time = datetime.datetime.strptime( \
        "2020-04-10 08:00:00", "%Y-%m-%d %H:%M:%S").timestamp()
    # startup_time = time.time()

    timing = TimingMHE(startup_time = startup_time)
    ambient = Ambient(timing)

    ambient.update()

    plt.figure()
    ax1 = plt.gca()

    for c in ["I_vtsc", "I_fpsc", "I_r_dir", "I_r_diff"]:
        ax1.plot(ambient.time_grid, ambient.c_data[:,ambient.c_index[c]], label = c)
    
    ax1.legend(loc = "best")

    ax2 = ax1.twinx()
    ax2.plot(ambient.time_grid, ambient.c_data[:,ambient.c_index["T_amb"]], label = "T_amb")
    ax2.legend(loc = "best")

    plt.show()

