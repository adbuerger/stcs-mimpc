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

import pandas as pd
from system import System

import logging

logger = logging.getLogger(__name__)

class Measurement(System):

    @property
    def time_grid(self):

        return self._timing.time_grid


    @property
    def y_data(self):

        y_index = []

        for y in self.y_index:

            try:

                for j, idx in enumerate(self.y_index[y]):

                    y_index.append(y + "[" + str(j) + "]")

            except TypeError:

                y_index.append(y)

        try:

            return self._df_measurement[y_index].values

        except AttributeError:

            msg = "Measurements not available yet, call update() first."
            
            logging.error(msg)
            raise RuntimeError(msg)


    @property
    def b_data(self):

        try:

            return self._df_measurement[list(self.b_index.keys())].values

        except AttributeError:

            msg = "Measurements not available yet, call update() first."
            
            logging.error(msg)
            raise RuntimeError(msg)


    @property
    def u_data(self):

        try:

            return self._df_measurement[list(self.u_index.keys())].values

        except AttributeError:

            msg = "Measurements not available yet, call update() first."
            
            logging.error(msg)
            raise RuntimeError(msg)


    def _setup_timing(self, timing):

        self._timing = timing


    def __init__(self, timing):

        super().__init__()

        self._setup_timing(timing)


    def _get_measurements(self):

        '''
        Load some generic measurements for demonstrational purposes.
        '''

        self._df_measurement = pd.read_csv("data/measurement_sample.csv")


    def update(self):

        self._get_measurements()


if __name__ == "__main__":

    import time
    import datetime
    import matplotlib.pyplot as plt

    from timing import TimingMHE

    startup_time = datetime.datetime.strptime( \
        "2020-04-10 08:00:00", "%Y-%m-%d %H:%M:%S").timestamp()
    # startup_time = time.time()

    timing = TimingMHE(startup_time = startup_time)
    measurement = Measurement(timing=timing)

    measurement.update()

    plt.figure()
    ax1 = plt.gca()

    for y in ["T_fpsc_s", "T_vtsc_s"]:
        ax1.plot(measurement.time_grid, measurement.y_data[:,measurement.y_index[y]], label = y)
    
    ax1.legend(loc = "best")

    ax2 = ax1.twinx()

    for y in ["I_fpsc", "I_vtsc"]:
        ax2.plot(measurement.time_grid, measurement.y_data[:,measurement.y_index[y]], label = y)
    
    ax2.legend(loc = "best")

    plt.show()

