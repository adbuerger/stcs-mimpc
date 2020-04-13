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
import os
import copy

import multiprocessing as mp

from timing import TimingMPC as Timing
from state import State
from ambient import Ambient

from simulator import Simulator

from predictor import Predictor

from nlpsolver import NLPSolverRel, NLPSolverBin
from binapprox import BinaryApproximation

from control import Control

import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

if not os.path.exists("log"):
    os.makedirs("log")

fh = RotatingFileHandler('log/mpc.log', 
    maxBytes = 10 * 1024 * 1024,
    backupCount = 2,
    mode = "a")
fh.setLevel(logging.DEBUG)
fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(fh_formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(ch_formatter)
logger.addHandler(ch)


class MPC(object):

    _MAX_MPC_ITERATIONS = 1000

    def run(self):

        def send_controls(timing, solver):

            control = Control(timing=timing, previous_solver=solver)
            control.apply()


        def solve_short_term_problem(timing, previous_solver, queue_st):

            ambient = Ambient(timing=timing)
            ambient.update()

            state = State()
            state.update()

            predictor = Predictor(timing=timing, ambient=ambient, \
                state=state, previous_solver=previous_solver, \
                solver_name="predictor_bin_" \
                    + str(timing.grid_position_cursor))
            predictor.solve()

            timing.increment_grid_position_cursor()

            if timing.grid_position_cursor >= timing.N_short_term:

                timing.shift_time_grid()
                ambient = Ambient(timing=timing)
                ambient.update()

            nlpsolver_bin = NLPSolverBin( \
                timing=timing, ambient=ambient, \
                previous_solver=previous_solver, predictor=predictor, \
                solver_name = "nlpsolver_bin_" \
                    + str(timing.grid_position_cursor))
            nlpsolver_bin.set_solver_max_cpu_time(time_point_to_finish=\
                timing.time_points[timing.grid_position_cursor])
            nlpsolver_bin.solve()
            nlpsolver_bin.reduce_object_memory_size()

            timing.sleep_until_time_grid_point("solve_short_term_problem", \
                timing.grid_position_cursor)

            queue_st.put(nlpsolver_bin)

            send_controls(timing, nlpsolver_bin)

            nlpsolver_bin.save_results()


        def generate_initial_controls(timing, queue_st):

            ambient = Ambient(timing=timing)
            ambient.update()

            state = State()
            state.update()

            simulator = Simulator( \
                timing=timing, ambient=ambient, state=state)

            simulator.solve()

            queue_st.put(simulator)

            simulator.save_results()


        def solve_long_term_problem(timing, previous_solver, queue_lt):

            ambient = Ambient(timing=timing)
            ambient.update()

            state = State()
            state.update()

            predictor = Predictor(timing=timing, ambient=ambient, \
                state=state, previous_solver=previous_solver, \
                solver_name="predictor_rel")
            predictor.solve(n_steps=timing.N_short_term)

            timing_next_interval = copy.deepcopy(timing)
            timing_next_interval.shift_time_grid()

            ambient = Ambient(timing=timing_next_interval)
            ambient.update()

            nlpsolver_rel = NLPSolverRel( \
                timing=timing_next_interval, ambient=ambient, \
                previous_solver=previous_solver, predictor=predictor, \
                solver_name="nlpsolver_rel")
            nlpsolver_rel.solve()
            nlpsolver_rel.save_results()

            binapprox = BinaryApproximation( \
                timing=timing_next_interval, previous_solver=nlpsolver_rel, \
                predictor=predictor, solver_name="binapprox")
            binapprox.set_solver_max_cpu_time(time_point_to_finish=\
                timing_next_interval.time_points[0])
            binapprox.solve()
            binapprox.save_results()

            timing.increment_grid_position_cursor(n_steps = timing.N_short_term - 1)
            timing.sleep_until_grid_position_cursor_time_grid_point("solve_long_term_problem")

            queue_lt.put((timing_next_interval, binapprox))

        timing = Timing(startup_time=time.time())

        queue_lt = mp.Queue()
        queue_st = mp.Queue()

        p = mp.Process(target = generate_initial_controls, args = (timing, queue_st))
        p.start()


        while timing.mpc_iteration_count < self._MAX_MPC_ITERATIONS:

            previous_solver_st = queue_st.get()

            timing.define_initial_switch_positions(previous_solver_st)
            
            p = mp.Process(target=solve_long_term_problem, \
                args=(timing, previous_solver_st, queue_lt))
            p.start()


            for k in range(timing.N_short_term-1):

                p = mp.Process(target=solve_short_term_problem, \
                    args=(timing, previous_solver_st, queue_st))
                p.start()

                timing.increment_grid_position_cursor()

                previous_solver_st = queue_st.get()


            # Retrieve results of long-term optimization
            
            timing_next_interval, previous_solver_lt = queue_lt.get() 

            p = mp.Process(target=solve_short_term_problem, \
                args=(timing, previous_solver_lt, queue_st))
            p.start()

            timing = copy.deepcopy(timing_next_interval)
            timing.increment_mpc_iteration_count()


if __name__ == "__main__":

    mpc = MPC()

    mpc.run()

