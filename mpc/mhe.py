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

from timing import TimingMHE
from measurement import Measurement
from ambient import Ambient

from nlpsolver import NLPSolverMHE

import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

startup_time = dt.datetime.strptime("2020-04-10 08:00:00", \
    "%Y-%m-%d %H:%M:%S").timestamp()
# startup_time = time.time()

timing = TimingMHE(startup_time = startup_time)

measurement = Measurement(timing=timing)
ambient = Ambient(timing=timing)

measurement.update()

nlpsolver_mhe = NLPSolverMHE(timing=timing, ambient=ambient, \
    measurement=measurement, solver_name="nlpsolver_mhe")

while True:

    measurement.update()
    ambient.update()

    nlpsolver_mhe.solve()
    nlpsolver_mhe.publish()
    nlpsolver_mhe.save_results()

    timing.shift_time_grid()
    timing.sleep_until_next_time_grid_point()
    
