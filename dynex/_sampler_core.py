"""
Dynex SDK (beta) Neuromorphic Computing Library
Copyright (c) 2021-2026, Dynex Developers

All rights reserved.

1. Redistributions of source code must retain the above copyright notice, this list of
    conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list
   of conditions and the following disclaimer in the documentation and/or other
   materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be
   used to endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Core sampler implementation: _DynexSampler orchestrates LocalSolverMixin and
SolutionStreamingMixin, runs the sampling loop and returns dimod SampleSets.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import os
import queue
import secrets
import subprocess
import time
import zlib
from dataclasses import dataclass, field
from typing import Optional

import dimod
import neal
import numpy as np
from IPython.core.display_functions import clear_output
from tabulate import tabulate

try:
    import grpc  # type: ignore
except ModuleNotFoundError:
    grpc = None  # type: ignore

try:
    import zstandard as zstd  # noqa: F401
except ModuleNotFoundError:
    zstd = None  # type: ignore

from dynex._local_solver import LocalSolverMixin
from dynex._streaming import SolutionStreamingMixin
from dynex.api import DynexAPI
from dynex.config import DynexConfig
from dynex.exceptions import DynexJobError, DynexModelError, DynexValidationError
from dynex.models import BQM


@dataclass
class SamplingTiming:
    start: float = field(default_factory=time.time)
    job_created: Optional[float] = None
    first_shot: Optional[float] = None
    all_shots: Optional[float] = None
    end: Optional[float] = None


def to_wcnf_string(clauses, num_variables, num_clauses):
    """Convert clauses to WCNF string format."""
    line = "p wcnf %d %d\n" % (num_variables, num_clauses)
    for clause in clauses:
        line += " ".join(str(int(lit)) for lit in clause) + " 0\n"
    return line


def _cqm_invert_safe(inverter, sample):
    """Safe CQM inversion that avoids dimod's int8 overflow for large integer variables."""
    from dimod import Vartype

    new = {}
    for v, vartype in inverter._binary.items():
        if vartype is Vartype.BINARY:
            new[v] = int(sample[v])
        elif vartype is Vartype.SPIN:
            new[v] = 2 * int(sample[v]) - 1
    for v, bqm in inverter._integers.items():
        new[v] = sum(int(sample[u]) * u[1] for u in bqm.variables)
    return new


class _DynexSampler(SolutionStreamingMixin, LocalSolverMixin):
    """Private sampler implementation - used internally by DynexSampler."""

    num_retries: int = 10

    def __init__(
        self,
        model,
        logging=True,
        description: Optional[str] = None,
        test=False,
        bnb=True,
        filename_override="",
        config: Optional[DynexConfig] = None,
        preserve_solutions: Optional[bool] = None,
        job_metadata: Optional[dict] = None,
    ):
        if model.type not in ("cnf", "wcnf", "qasm"):
            raise DynexModelError(f"Unsupported model type: {model.type}")

        self.config = config if config is not None else DynexConfig()
        self.description = description if description is not None else self.config.default_description
        self.api = DynexAPI(config=self.config, logging=logging)
        self.logger = self.config.logger
        self.job_metadata = job_metadata

        self.filepath = "tmp/"
        self.filepath_full = os.getcwd() + "/tmp"
        self.current_job_id = None

        # Initialise gRPC streaming and solution cache state
        self._init_streaming_state()

        self.preserve_solutions = (
            preserve_solutions if preserve_solutions is not None else self.config.preserve_solutions
        )
        self.use_notebook_output = self.config.use_notebook_output
        self.timeout = self.config.default_timeout

        self.solver_path = self.config.solver_path
        self.bnb = bnb

        multi_model_mode = False
        if isinstance(model, list):
            if not self.config.mainnet:
                raise DynexValidationError("Multi model parallel sampling is only supported in network mode")
            multi_model_mode = True

        self.multi_model_mode = multi_model_mode

        if not multi_model_mode:
            if len(filename_override) > 0:
                if filename_override.endswith(".dnx"):
                    self.filename = filename_override
                else:
                    self.filename = filename_override + ".dnx"
            else:
                self.filename = secrets.token_hex(16) + ".dnx"

            self.logging = logging
            self.type_str = model.type_str
            self.wcnf_offset = model.wcnf_offset
            self.precision = model.precision

            if model.type == "wcnf":
                self.num_variables = model.bqm.num_variables
                self.num_clauses = len(model.bqm.to_qubo()[0])
                self.clauses = model.bqm.to_qubo()
                self.var_mappings = model.var_mappings
                self.precision = model.precision
                if not self.config.mainnet:
                    self._save_wcnf(
                        self.clauses,
                        self.filepath + self.filename,
                        self.num_variables,
                        self.num_clauses,
                        self.var_mappings,
                    )

            elif model.type == "qasm":
                self.clauses = [0, -9999999999]
                self.num_variables = None
                self.num_clauses = None
                self.var_mappings = None
                self.precision = None

            self.type = model.type
            self.assignments = {}
            self.dimod_assignments = {}
            self.bqm = model.bqm
            self.model = model

        else:
            _filename = []
            _type_str = []
            _clauses = []
            _num_clauses = []
            _num_variables = []
            _var_mappings = []
            _precision = []
            _type = []
            _assignments = []
            _dimod_assignments = []
            _bqm = []
            _model = []
            for m in model:
                _filename.append(secrets.token_hex(16) + ".dnx")
                _type_str.append(m.type)
                if m.type == "wcnf":
                    _num_variables.append(m.bqm.num_variables)
                    _num_clauses.append(len(m.bqm.to_qubo()[0]))
                    _clauses.append(m.bqm.to_qubo())
                    _var_mappings.append(m.var_mappings)
                    _precision.append(m.precision)
                    if not self.config.mainnet:
                        self._save_wcnf(
                            _clauses[-1],
                            self.filepath + _filename[-1],
                            _num_variables[-1],
                            _num_clauses[-1],
                            _var_mappings[-1],
                        )
                _type.append(m.type)
                _assignments.append({})
                _dimod_assignments.append({})
                _bqm.append(m.bqm)
                _model.append(m)
            self.filename = _filename
            self.type_str = _type_str
            self.clauses = _clauses
            self.num_clauses = _num_clauses
            self.num_variables = _num_variables
            self.var_mappings = _var_mappings
            self.precision = _precision
            self.type = _type
            self.assignments = _assignments
            self.dimod_assignments = _dimod_assignments
            self.bqm = _bqm
            self.model = _model
            self.wcnf_offset = _model.wcnf_offset
            self.precision = _model.precision
            self.logging = logging

        if self.logging:
            self.logger.info("Sampler initialised")

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def is_cluster_mode(self) -> bool:
        """True when sampling multiple models in parallel (cluster mode)."""
        return isinstance(self.clauses, list)

    # ------------------------------------------------------------------ #
    # Logging                                                              #
    # ------------------------------------------------------------------ #

    def _log_debug(self, message: str) -> None:
        if self.logger and self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(message)

    # ------------------------------------------------------------------ #
    # Energy calculation                                                   #
    # ------------------------------------------------------------------ #

    def _energy(self, sample, mapping=True) -> tuple:
        """Calculate energy and loc from dimod sample or raw solution."""
        wcnf_vars = []
        if mapping:
            for v in sample:
                if v in self.model.var_mappings:
                    v_mapped = self.model.var_mappings[v]
                else:
                    v_mapped = v
                wcnf_vars.append(sample[v_mapped])
        else:
            for v in sample:
                if v > 0:
                    wcnf_vars.append(1)
                else:
                    wcnf_vars.append(0)

        loc = 0
        energy = 0.0
        for clause in self.model.clauses:
            if len(clause) == 2:
                w = clause[0]
                i = int(abs(clause[1]))
                i_dir = np.sign(clause[1])
                if i_dir == -1:
                    i_dir = 0
                i_assign = wcnf_vars[i - 1]
                if i_dir != i_assign:
                    loc += 1
                    energy += w
            else:
                w = clause[0]
                i = int(abs(clause[1]))
                i_dir = np.sign(clause[1])
                if i_dir == -1:
                    i_dir = 0
                i_assign = wcnf_vars[i - 1]
                j = int(abs(clause[2]))
                j_dir = np.sign(clause[2])
                if j_dir == -1:
                    j_dir = 0
                j_assign = wcnf_vars[j - 1]
                if (i_dir != i_assign) and (j_dir != j_assign):
                    loc += 1
                    energy += w
        return loc, energy

    # ------------------------------------------------------------------ #
    # Job control                                                          #
    # ------------------------------------------------------------------ #

    def _try_cancel_job(self, job_id, mainnet: bool) -> None:
        if not mainnet or job_id is None:
            return
        try:
            self.api.cancel_job_api(job_id)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def _clean(self) -> None:
        """Cleanup after sampling - removes unused solution files."""
        if self.config.mainnet:
            self.list_files_with_text_local()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Cleanup on sampler exit."""
        self.logger.info("Sampler exit")

    def _update(self, model, logging=True) -> None:
        """Update sampler with new model without creating new sampler."""
        self.logging = logging
        self.filename = secrets.token_hex(16) + ".dnx"

        if model.type == "wcnf":
            self.num_variables = model.bqm.num_variables
            self.num_clauses = len(model.bqm.to_qubo()[0])
            self.clauses = model.to_qubo()
            self.var_mappings = model.var_mappings
            self.precision = model.precision
            if not self.config.mainnet:
                self._save_wcnf(
                    self.clauses,
                    self.filepath + self.filename,
                    self.num_variables,
                    self.num_clauses,
                    self.var_mappings,
                )

        self.type = model.type
        self.assignments = {}
        self.dimod_assignments = {}
        self.bqm = model.bqm

    # ------------------------------------------------------------------ #
    # Utilities                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _convert(a) -> dict:
        """Convert flat list to dict (key-value pairs)."""
        it = iter(a)
        return dict(zip(it, it))

    def _print(self) -> None:
        """Print sampler summary."""
        self.logger.info("{DynexSampler object}")
        self.logger.info(f"network_mode? {self.config.mainnet}")
        self.logger.info(f"logging? {self.logging}")
        self.logger.info(f"tmp filename: {self.filepath + self.filename}")
        self.logger.info(f"model type: {self.type_str}")
        self.logger.info(f"num variables: {self.num_variables}")
        self.logger.info(f"num clauses: {self.num_clauses}")
        self.logger.info("configuration loaded")

    def _sample_to_assignments(self, lowest_set) -> dict:
        """Convert voltage list (-1/+1) to binary sample dict (0/1)."""
        sample = {}
        i = 0
        for var in self.var_mappings:
            sample[var] = 1
            if i < len(lowest_set) and float(lowest_set[i]) < 0:
                sample[var] = 0
            i += 1
        return sample

    def __repr__(self) -> str:
        return f"<_DynexSampler job={self.current_job_id!r} " f"type={self.type!r} vars={self.num_variables}>"

    # ------------------------------------------------------------------ #
    # Public sampling entry point                                          #
    # ------------------------------------------------------------------ #

    def sample(
        self,
        num_reads=32,
        annealing_time=10,
        switchfraction=0.0,
        alpha=20,
        beta=20,
        gamma=1,
        delta=1,
        epsilon=1,
        zeta=1,
        minimum_stepsize=0.05,
        debugging=False,
        block_fee=0,
        shots=1,
        rank=1,
        preprocess=False,
        qpu_max_coeff=9.0,
    ) -> "dimod.SampleSet":
        """Main sampling entry point - delegates to _sample."""
        retval = {}

        # In a malleable environment, a worker may occasionally submit an inconsistent solution file.
        # This routine samples up to NUM_RETRIES (10) times before giving up.

        self.expected_shots = shots

        for i in range(0, self.num_retries):
            retval = self._sample(
                num_reads=num_reads,
                annealing_time=annealing_time,
                switchfraction=switchfraction,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                delta=delta,
                epsilon=epsilon,
                zeta=zeta,
                minimum_stepsize=minimum_stepsize,
                debugging=debugging,
                block_fee=block_fee,
                shots=shots,
                rank=rank,
                preprocess=preprocess,
                qpu_max_coeff=qpu_max_coeff,
            )
            if len(retval) > 0:
                break

            self.logger.info(f"No valid sample result found. Resampling... {i + 1} / {self.num_retries}")
            self.filename = secrets.token_hex(16) + ".dnx"
            if self.type == "wcnf" and not self.config.mainnet:
                self._save_wcnf(
                    self.clauses,
                    self.filepath + self.filename,
                    self.num_variables,
                    self.num_clauses,
                    self.model.var_mappings,
                )

        return retval

    # ------------------------------------------------------------------ #
    # Core sampling implementation                                         #
    # ------------------------------------------------------------------ #

    def _sample(
        self,
        num_reads=32,
        annealing_time=10,
        switchfraction=0.0,
        alpha=20,
        beta=20,
        gamma=1,
        delta=1,
        epsilon=1,
        zeta=1,
        minimum_stepsize=0.05,
        debugging=False,
        block_fee=0,
        shots=1,
        rank=1,
        preprocess=False,
        qpu_max_coeff=9.0,
    ):
        """Perform sampling - upload job, wait for solutions, parse results."""
        self._timing = SamplingTiming()

        if self.multi_model_mode is True:
            raise DynexJobError("Multi-model parallel sampling is not implemented yet")

        MAX_ANNEALING_TIME_QPU = 10000
        compute_backend = getattr(self.config, "compute_backend", None)
        is_qpu = compute_backend and (compute_backend == "qpu" or compute_backend == "QPU")

        if is_qpu:
            qpu_model = self.config.qpu_model

            if self.job_metadata is None:
                self.job_metadata = {}
            self.job_metadata["qpu_model"] = str(qpu_model)

            if self.logging:
                self.logger.info(f"Apollo QPU chip: {qpu_model}")
                is_circuit_bqm = self.job_metadata and self.job_metadata.get("type") == "qasm"
                if not is_circuit_bqm and self.num_variables is not None:
                    self.logger.info(f"Problem: {self.num_variables} qubits, {self.num_clauses} gates")
                self.logger.info(f"Settings: num_reads={num_reads}, shots={shots}, annealing_time={annealing_time}")

                if self.num_variables and annealing_time < 100 and self.num_variables > 100:
                    self.logger.warning(
                        f"annealing_time={annealing_time} might be short for {self.num_variables} qubits "
                        f"(recommended: >=100 for problems >100 qubits)"
                    )
                if num_reads > 50:
                    self.logger.warning(f"num_reads={num_reads} is very high, consider reducing for faster testing")
                if shots > 20:
                    self.logger.warning(f"shots={shots} is very high, consider reducing for faster testing")

            if annealing_time > MAX_ANNEALING_TIME_QPU:
                if self.logging:
                    self.logger.warning(
                        f"annealing_time ({annealing_time}) exceeds Apollo QPU limit ({MAX_ANNEALING_TIME_QPU}), "
                        f"capping to {MAX_ANNEALING_TIME_QPU}"
                    )
                annealing_time = MAX_ANNEALING_TIME_QPU

            is_circuit_bqm = self.job_metadata and self.job_metadata.get("type") == "qasm"

            if self.bqm and not is_circuit_bqm:
                from .preprocessing import scale_bqm_to_range

                max_abs = 0.0
                for coeff in self.bqm.linear.values():
                    max_abs = max(max_abs, abs(float(coeff)))
                for coeff in self.bqm.quadratic.values():
                    max_abs = max(max_abs, abs(float(coeff)))

                if max_abs > qpu_max_coeff:
                    if self.logging:
                        self.logger.info(f"Auto-scaling BQM for QPU: max_abs_coeff={max_abs:.2f} > {qpu_max_coeff}")
                    scaled_bqm, scaling_factor = scale_bqm_to_range(self.bqm, max_abs_coeff=qpu_max_coeff)
                    self.bqm = scaled_bqm
                    self._bqm_scaling_factor = scaling_factor
                    if self.logging:
                        self.logger.info(f"BQM scaled by factor {scaling_factor:.6f} for optimal QPU performance")
                else:
                    self._bqm_scaling_factor = 1.0
                    if self.logging:
                        self.logger.info(f"BQM coefficients within range (max={max_abs:.2f}), no scaling needed")

        mainnet = self.config.mainnet
        price_per_block = 0
        self.cnt_solutions = 0

        dimod_sample = []
        job_id = None

        if self.bqm and not preprocess:
            self.model.wcnf_offset = self.bqm.offset
            self.model.precision = 1

        if self.bqm and preprocess and compute_backend not in ("qpu", "QPU"):
            sampler_sa = neal.SimulatedAnnealingSampler()
            sampleset = []
            start_time = time.time()

            for shot in range(0, shots):
                _sampleset = sampler_sa.sample(self.bqm, num_reads=num_reads, num_sweeps=annealing_time)
                if not sampleset:
                    sampleset = _sampleset
                else:
                    sampleset = dimod.concatenate([sampleset, _sampleset])

            end_time = time.time()
            elapsed_time = (end_time - start_time) * 100

            self.logger.info(f"Preprocessed with energy {sampleset.first.energy} offset={self.bqm.offset}")
            if sampleset.first.energy <= 0:
                if self.logging:
                    self.logger.info(f"Finished read after {elapsed_time} seconds")
                table = [
                    [
                        "DYNEXJOB",
                        "QUBITS",
                        "GATES",
                        "NUM_READS",
                        "SHOTS",
                        "ANN.TIME",
                        "ELAPSED",
                        "WORKERS",
                        "GROUND STATE",
                    ]
                ]
                table.append(
                    [
                        "-1",
                        self.num_variables,
                        self.num_clauses,
                        num_reads,
                        1,
                        annealing_time,
                        f"{elapsed_time:.2f}s",
                        "PREPROCESSED",
                        sampleset.first.energy,
                    ]
                )
                ta = tabulate(table, headers="firstrow", tablefmt="rounded_grid", floatfmt=".2f")
                self.logger.info(ta + "\n")
                return sampleset

            dimod_sample = [sampleset.first.sample]

        try:
            if mainnet:
                job_metadata = self.job_metadata
                if job_metadata is None and self.type == "qasm":
                    job_metadata = {"type": "qasm"}

                params = {
                    "sampler": self,
                    "annealing_time": annealing_time,
                    "switchfraction": switchfraction,
                    "num_reads": num_reads,
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "delta": delta,
                    "epsilon": epsilon,
                    "zeta": zeta,
                    "minimum_stepsize": minimum_stepsize,
                    "block_fee": block_fee,
                    "shots": shots,
                    "rank": rank,
                    "target_energy": 0.0 - self.clauses[1],
                    "job_metadata": job_metadata,
                }

                job_id, self.filename, price_per_block, qasm = self.api.create_job_api_proto(
                    **params, debugging=debugging
                )
                self._timing.job_created = time.time()
                self._reset_grpc_subscription()
                self.current_job_id = job_id
                self._downloaded_solutions.clear()
                price_per_block = price_per_block / 1000000000

                if self.type == "qasm":
                    if qasm is None:
                        self.logger.error("QASM data is None from backend. Backend did not return qasm_json.")
                        self.logger.error(
                            "This indicates the backend doesn't support QASM conversion or the converter failed."
                        )
                        raise DynexJobError("QASM data is None. Backend may not support QASM processing.")
                    _data = qasm
                    if not isinstance(_data, dict) or "feed_dict" not in _data or "model" not in _data:
                        self.logger.error(f"Invalid QASM data format: {type(_data)}")
                        raise DynexJobError(
                            "Invalid QASM data format. Expected dict with 'feed_dict' and 'model' keys."
                        )
                    _feed_dict = _data["feed_dict"]
                    _model = _data["model"]
                    if debugging:
                        self.logger.info(f"feed_dict: {_feed_dict}")
                        self.logger.info(f"model: {_model}")
                    q = zlib.decompress(bytearray.fromhex(_model["q"]))
                    q = str(q)[2:-1]
                    offset = float(_model["offset"])
                    bqm = dimod.BinaryQuadraticModel.from_qubo(ast.literal_eval(q), offset)

                    _model = BQM(bqm)
                    self.bqm = bqm
                    self.num_variables = _model.bqm.num_variables
                    self.num_clauses = len(_model.bqm.to_qubo()[0])
                    self.clauses = _model.bqm.to_qubo()
                    self.var_mappings = _model.var_mappings
                    self.precision = _model.precision
                    if self.config.debug_save_solutions:
                        self._save_wcnf(
                            self.clauses,
                            self.filepath + self.filename,
                            self.num_variables,
                            self.num_clauses,
                            self.var_mappings,
                        )
                    self.model.clauses = self.clauses
                    self.model.num_variables = self.num_variables
                    self.model.num_clauses = self.num_clauses
                    self.model.var_mappings = self.var_mappings
                    self.model.precision = self.precision
                    if self.logging and is_qpu:
                        self.logger.info(
                            f"Problem: {self.num_variables} qubits, {self.num_clauses} gates (Circuit BQM from QASM)"
                        )

                if self.logging:
                    if not is_qpu:
                        self.logger.info(f"Problem: {self.num_variables} qubits, {self.num_clauses} gates")
                        self.logger.info(
                            f"Settings: num_reads={num_reads}, shots={shots}, annealing_time={annealing_time}"
                        )
                        if annealing_time < 100 and self.num_variables > 100:
                            self.logger.warning(
                                f"annealing_time={annealing_time} might be short for {self.num_variables} qubits "
                                f"(recommended: >=100 for problems >100 qubits)"
                            )
                        if num_reads > 50:
                            self.logger.warning(
                                f"num_reads={num_reads} is very high, consider reducing for faster testing"
                            )
                        if shots > 20:
                            self.logger.warning(f"shots={shots} is very high, consider reducing for faster testing")
                    self.logger.info("Starting job...")
            else:
                if self.type == "qasm":
                    command = (
                        "python3 dynex_circuit_backend.py --mainnet False --file "
                        + self.model.qasm_filepath
                        + self.model.qasm_filename
                    )
                    if debugging:
                        command = command + " --debugging True"
                        process = subprocess.Popen(
                            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
                        )
                        if self.logging:
                            for line in iter(process.stdout.readline, ""):
                                if line.strip():
                                    self.logger.debug(f"[QASM] {line.rstrip()}")
                        process.wait()
                    else:
                        if self.logging:
                            self.logger.info("Waiting for reads...")
                        process = subprocess.Popen(
                            command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                        )
                        process.wait()
                    f = open(self.model.qasm_filepath + self.model.qasm_filename + ".model", "r", encoding="utf-8")
                    _data = json.load(f)
                    _feed_dict = _data["feed_dict"]
                    _model = _data["model"]
                    if debugging:
                        self.logger.debug("feed_dict:")
                        self.logger.debug(_feed_dict)
                        self.logger.debug("model:")
                        self.logger.debug(_model)
                    f.close()
                    q = zlib.decompress(bytearray.fromhex(_model["q"]))
                    q = str(q)[2:-1]
                    offset = _model["offset"]
                    bqm = dimod.BinaryQuadraticModel.from_qubo(ast.literal_eval(q), offset)

                    _model = BQM(bqm)
                    self.bqm = bqm
                    self.num_variables = _model.bqm.num_variables
                    self.num_clauses = len(_model.bqm.to_qubo()[0])
                    self.clauses = _model.bqm.to_qubo()
                    self.var_mappings = _model.var_mappings
                    self.precision = _model.precision
                    self._save_wcnf(
                        self.clauses,
                        self.filepath + self.filename,
                        self.num_variables,
                        self.num_clauses,
                        self.var_mappings,
                    )

                job_id = ""

                bnb_binary = self.solver_path + "dynex-testnet-bnb"
                if self.bnb and os.path.exists(bnb_binary):
                    command = bnb_binary + " " + self.filepath_full + "/" + self.filename
                else:
                    population_size = num_reads
                    if rank > population_size:
                        raise DynexValidationError(
                            f"Rank must be equal to population size! Shots:{rank} Population:{population_size}"
                        )
                    command = self.solver_path + "dynexcore"
                    command += " file=" + self.filepath_full + "/" + self.filename
                    command += " num_steps=" + str(annealing_time)
                    command += " population_size=" + str(num_reads)
                    command += " max_iterations=" + str(num_reads)
                    command += " target_energy=" + str(0.0 - self.clauses[1])
                    command += " init_dt=" + str(minimum_stepsize)
                    command += " cpu_threads=4"
                    command += " shots=" + str(rank)

                if debugging:
                    self.logger.debug(f"Solver command: {command}")
                    self.logger.debug(f"Working directory: {self.filepath_full}")
                    self.logger.debug(f"Problem file: {self.filepath_full}/{self.filename}")

                for shot in range(0, shots):
                    if debugging:
                        self.logger.debug(f"Starting solver shot {shot + 1}/{shots}")
                        process = subprocess.Popen(
                            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
                        )
                        if self.logging:
                            for line in iter(process.stdout.readline, ""):
                                if line.strip():
                                    self.logger.debug(f"[SOLVER] {line.rstrip()}")
                        returncode = process.wait()
                        self.logger.debug(f"Solver finished with return code: {returncode}")
                    else:
                        process = subprocess.Popen(
                            command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                        )
                        process.wait()

                    self.add_salt_local()
                    if debugging:
                        self.logger.debug(f"Salt added for shot {shot + 1}")

                if self.logging:
                    self.logger.info(f"Completed {shots} solver iterations")

            t = time.process_time()
            t_start = time.time()
            finished = False
            runupdated = False
            cnt_workers = 0
            compute_backend = getattr(self.config, "compute_backend", None)
            max_wait_time = getattr(self, "timeout", 300.0)

            if mainnet and not debugging and self.use_notebook_output:
                clear_output(wait=True)
                table = [
                    [
                        "DYNEXJOB",
                        "QUBITS",
                        "GATES",
                        "NUM_READS",
                        "SHOTS",
                        "ANN.TIME",
                        "ELAPSED",
                        "WORKERS",
                        "GROUND STATE",
                    ]
                ]
                table.append(
                    [
                        "",
                        self.num_variables,
                        self.num_clauses,
                        num_reads,
                        shots,
                        annealing_time,
                        "",
                        "*** WAITING ***",
                        "",
                    ]
                )
                ta = tabulate(table, headers="firstrow", tablefmt="rounded_grid", floatfmt=".2f")
                if self.logging:
                    self.logger.info("\n" + ta + "\n")

            table_update_counter = 0
            prev_files_count = 0

            while not finished:
                if self._job_error:
                    error_msg = f"Job failed: {self._job_error}"
                    if self.logging:
                        self.logger.error(f"{error_msg}")
                    raise DynexJobError(error_msg)

                elapsed_real_time = time.time() - t_start
                if elapsed_real_time > max_wait_time:
                    solutions_count = len(self._downloaded_solutions)
                    if solutions_count == 0:
                        error_msg = (
                            f"Timeout waiting for solutions after {max_wait_time:.0f} seconds. "
                            f"No solutions received from compute backend. "
                            f"This may indicate a failure in the backend (Modal/Apollo API). "
                            f"Check logs for errors like 401 Unauthorized, SSL errors, or network issues."
                        )
                        if self.logging:
                            self.logger.error(f"{error_msg}")
                        raise DynexJobError(error_msg)
                    else:
                        if self.logging:
                            self.logger.warning(
                                f"Timeout after {max_wait_time:.0f}s, but received "
                                f"{solutions_count} solution(s). Continuing..."
                            )
                    finished = True
                    break

                total_chips = 0
                total_steps = 0
                lowest_energy = 1.7976931348623158e308
                lowest_loc = 1.7976931348623158e308

                if mainnet:
                    try:
                        self.list_files_with_text()
                    except Exception:
                        pass

                files = self.list_files_with_text_local()
                cnt_workers = len(files)

                elapsed_int = int(elapsed_real_time)
                if (
                    self.logging
                    and cnt_workers == 0
                    and elapsed_int > 0
                    and elapsed_int % 10 == 0
                    and elapsed_int != getattr(self, "_last_log_time", -1)
                ):
                    self._last_log_time = elapsed_int
                    self.logger.info(
                        f"Still waiting for solutions... elapsed={elapsed_int}s, timeout={int(max_wait_time)}s, "
                        f"job_id={self.current_job_id}, files={len(files)}, "
                        f"downloaded={len(self._downloaded_solutions)}"
                    )

                if debugging and not mainnet and len(files) != prev_files_count:
                    self.logger.debug(
                        f"Local mode: found {len(files)} solution files, "
                        f"cnt_solutions={self.cnt_solutions}, shots={shots}"
                    )

                solutions_count = len(self._downloaded_solutions)
                if mainnet:
                    self.cnt_solutions = solutions_count
                else:
                    self.cnt_solutions += len(files)

                if mainnet:
                    if solutions_count >= shots:
                        self._log_debug(
                            f"Exit condition met: solutions_count={solutions_count} >= shots={shots}, finishing..."
                        )
                        if self._timing.all_shots is None:
                            self._timing.all_shots = time.time()
                        finished = True
                    else:
                        if (
                            elapsed_int > 0
                            and elapsed_int % 5 == 0
                            and elapsed_int != getattr(self, "_last_wait_log_time", -1)
                        ):
                            self._last_wait_log_time = elapsed_int
                            debug_msg = (
                                f"Waiting for solutions: solutions_count={solutions_count} < shots={shots}, "
                                f"files={len(files)}, downloaded={len(self._downloaded_solutions)}, "
                                f"queue_size={self._grpc_solution_queue.qsize()}, job_id={self.current_job_id}"
                            )
                            self._log_debug(debug_msg)
                else:
                    if self.cnt_solutions >= shots:
                        finished = True

                prev_files_count = len(files)

                if not files:
                    if finished:
                        break
                    time.sleep(0.5)
                    continue

                for file in files:
                    chips, steps, loc, energy = self._get_solution_metrics(file)

                    chip_val = chips if isinstance(chips, int) else self._coerce_int(chips)
                    step_val = steps if isinstance(steps, int) else self._coerce_int(steps)
                    loc_val = loc if isinstance(loc, int) else self._coerce_int(loc)
                    energy_val = energy if isinstance(energy, float) else self._coerce_float(energy)

                    if chip_val and chip_val > 0:
                        total_chips += chip_val
                    if step_val and step_val > 0:
                        total_steps = step_val
                    if energy_val is not None and energy_val < lowest_energy:
                        lowest_energy = energy_val
                    if loc_val is not None and loc_val < lowest_loc:
                        lowest_loc = loc_val

                if finished:
                    break

                details = ""
                display_energy = lowest_energy if math.isfinite(lowest_energy) else 0.0
                table_update_counter += 1
                should_show_table = (
                    (table_update_counter == 1) or finished or (cnt_workers > 0 and table_update_counter % 10 == 0)
                )

                if self.logging and should_show_table and mainnet:
                    if not debugging and self.use_notebook_output:
                        clear_output(wait=True)

                    details = "*** WAITING FOR WORKERS ***"
                    table = [
                        [
                            "DYNEXJOB",
                            "QUBITS",
                            "GATES",
                            "NUM_READS",
                            "SHOTS",
                            "ANN.TIME",
                            "ELAPSED",
                            "WORKERS",
                            "GROUND STATE",
                        ]
                    ]
                    if cnt_workers < 1:
                        table.append(
                            [
                                job_id,
                                self.num_variables,
                                self.num_clauses,
                                num_reads,
                                shots,
                                annealing_time,
                                f"{elapsed_real_time:.0f}s",
                                "*** WAITING ***",
                                0,
                            ]
                        )
                    else:
                        elapsed_time = time.process_time() - t
                        table.append(
                            [
                                job_id,
                                self.num_variables,
                                self.num_clauses,
                                num_reads,
                                shots,
                                annealing_time,
                                f"{elapsed_time:.2f}s",
                                cnt_workers,
                                (display_energy + self.model.wcnf_offset) * self.model.precision,
                            ]
                        )
                    ta = tabulate(table, headers="firstrow", tablefmt="rounded_grid", floatfmt=".2f")
                    self.logger.info(f"\n{ta}\n{details}")

                    if not runupdated and mainnet:
                        self.api.update_job_api(job_id)
                        runupdated = True

                if not finished:
                    time.sleep(0.5)

            summary_files = self.list_files_with_text_local()
            summary_lowest_loc = float("inf")
            summary_lowest_energy = float("inf")
            summary_total_chips = 0
            summary_total_steps = 0
            for file in summary_files:
                chips, steps, loc, energy = self._get_solution_metrics(file)
                if chips:
                    summary_total_chips += chips
                if steps:
                    summary_total_steps = steps
                if energy is not None and energy < summary_lowest_energy:
                    summary_lowest_energy = energy
                if loc is not None and loc < summary_lowest_loc:
                    summary_lowest_loc = loc

            lowest_energy = summary_lowest_energy
            lowest_loc = summary_lowest_loc
            total_chips = summary_total_chips
            total_steps = summary_total_steps

            if mainnet:
                final_loc = int(summary_lowest_loc) if math.isfinite(summary_lowest_loc) else 0
                final_energy = float(summary_lowest_energy) if math.isfinite(summary_lowest_energy) else 0.0
                self.api.finish_job_api(job_id, final_loc, final_energy)

            if cnt_workers > 0 and self.logging:
                if mainnet and not debugging and self.use_notebook_output:
                    clear_output(wait=True)
                details = ""
                display_energy = lowest_energy if math.isfinite(lowest_energy) else 0.0
                if mainnet:
                    details = "*** WAITING FOR WORKERS ***"

                elapsed_time = time.process_time() - t
                if mainnet:
                    table = [
                        [
                            "DYNEXJOB",
                            "QUBITS",
                            "GATES",
                            "NUM_READS",
                            "SHOTS",
                            "ANN.TIME",
                            "ELAPSED",
                            "WORKERS",
                            "GROUND STATE",
                        ]
                    ]
                    table.append(
                        [
                            job_id,
                            self.num_variables,
                            self.num_clauses,
                            num_reads,
                            shots,
                            annealing_time,
                            f"{elapsed_time:.2f}s",
                            cnt_workers,
                            (display_energy + self.model.wcnf_offset) * self.model.precision,
                        ]
                    )
                    ta = tabulate(table, headers="firstrow", tablefmt="rounded_grid", floatfmt=".2f")
                    if self.logging:
                        self.logger.info(f"\n{ta}\n{details}")

            self._timing.end = time.time()
            elapsed_time = time.process_time() - t
            elapsed_time *= 100

            if self.logging:
                if mainnet:
                    total_time = self._timing.end - self._timing.start
                    upload_time = (self._timing.job_created - self._timing.start) if self._timing.job_created else 0
                    first_shot_time = (
                        (self._timing.first_shot - self._timing.job_created)
                        if self._timing.first_shot and self._timing.job_created
                        else 0
                    )
                    compute_time = (
                        (self._timing.all_shots - self._timing.job_created)
                        if self._timing.all_shots and self._timing.job_created
                        else 0
                    )
                    download_time = (self._timing.end - self._timing.all_shots) if self._timing.all_shots else 0

                    if shots > 1 and compute_time > 0:
                        avg_time_per_shot = compute_time / shots
                        self.logger.info(
                            f"Average time per shot: {avg_time_per_shot:.2f}s ({shots} shots in {compute_time:.2f}s)"
                        )

                    self.logger.info("Timing breakdown:")
                    self.logger.info(f"  Job upload:        {upload_time:.2f}s")
                    if first_shot_time > 0:
                        self.logger.info(f"  Time to 1st shot:  {first_shot_time:.2f}s")
                    self.logger.info(f"  Compute:  {compute_time:.2f}s")
                    self.logger.info(f"  Solution download: {download_time:.2f}s")
                    self.logger.info(f"  Total elapsed:     {total_time:.2f}s")

                self.logger.info(f"Finished read after {elapsed_time} seconds")

            sampleset = []
            lowest_energy = 1.7976931348623158e308
            lowest_loc = 1.7976931348623158e308
            total_chips = 0
            total_steps = 0
            lowest_set = []
            dimod_sample = []
            for file in files:
                chips, steps, loc, energy = self._get_solution_metrics(file)

                total_chips += chips if chips else 0
                total_steps = steps

                voltages = self.read_voltage_data(file, mainnet, rank)

                if self.type in ["wcnf", "qasm"] and voltages and len(voltages) > 0:
                    sampleset.append(
                        ["sample", voltages, "chips", chips, "steps", steps, "falsified softs", loc, "energy", energy]
                    )
                    if loc < lowest_loc:
                        lowest_loc = loc
                    if energy < lowest_energy:
                        lowest_energy = energy
                        lowest_set = voltages
                    usable_len = min(len(voltages), self.num_variables)
                    if usable_len < self.num_variables:
                        self._log_debug(
                            f"Voltage result shorter than expected solver=2 "
                            f"got={len(voltages)} expected={self.num_variables}"
                        )
                    dimodsample = {}
                    for var, value in enumerate(voltages[:usable_len]):
                        if var in self.var_mappings:
                            dimodsample[self.var_mappings[var]] = 1
                            if float(value) < 0:
                                dimodsample[self.var_mappings[var]] = 0
                        else:
                            dimodsample[var] = 1
                            if float(value) < 0:
                                dimodsample[var] = 0

                    dimod_sample.append(dimodsample)

            if self.type in ["wcnf", "qasm"]:
                sampleset.append(
                    [
                        "sample",
                        lowest_set,
                        "chips",
                        total_chips,
                        "steps",
                        total_steps,
                        "falsified softs",
                        lowest_loc,
                        "energy",
                        lowest_energy,
                    ]
                )

            min_required_vars = int(self.num_variables * 0.9) if self.num_variables else 0
            if (
                (self.type in ["wcnf", "qasm"])
                and len(lowest_set) >= min_required_vars
                and len(lowest_set) <= self.num_variables
            ):
                sample = {}
                i = 0
                for var in self.var_mappings:
                    if i < len(lowest_set):
                        sample[var] = 1
                        if float(lowest_set[i]) < 0:
                            sample[var] = 0
                        i = i + 1
                    else:
                        sample[var] = 0
                self.assignments = sample

                self.dimod_assignments = dimod.SampleSet.from_samples_bqm(dimod_sample, self.bqm)
            elif self.type in ["wcnf", "qasm"]:
                self.logger.warning(
                    "No valid sampleset returned from sampling. The job may have been cancelled or failed."
                )
                self.logger.warning(
                    f"lowest_set length ({len(lowest_set)}) does not match num_variables ({self.num_variables})"
                )
                self.dimod_assignments = dimod.SampleSet.from_samples_bqm([], self.bqm)

            if self.logging:
                self.logger.info(f"Sampleset ready with energy {self.dimod_assignments}")

            sampleset_clean = []
            for sample in sampleset:
                sample_dict = self._convert(sample)
                sampleset_clean.append(sample_dict)

            if (self.config.mainnet or self.config.remove_local_solutions) and not self.preserve_solutions:
                self.delete_local_files_by_prefix(self.filepath, self.filename)

        except (KeyboardInterrupt, Exception) as exc:
            self._try_cancel_job(job_id, mainnet)
            if isinstance(exc, KeyboardInterrupt):
                self.logger.error("Keyboard interrupt")
            raise
        finally:
            if self.config.mainnet:
                self._stop_grpc_subscription()
                self._grpc_solution_queue = queue.Queue()
                self._grpc_last_seq = 0
                self._grpc_subscription_bootstrap_done = False
                self._grpc_solution_meta = {}
                self._grpc_solution_stats = {}
                self._grpc_solution_remote = {}
                self._job_error = None
                self.current_job_id = None

        self.sampleset = sampleset_clean

        if self.model.type_str == "CQM":
            try:
                cqm_sample = self.model.invert(self.dimod_assignments.first.sample)
            except OverflowError:
                cqm_sample = _cqm_invert_safe(self.model.invert, self.dimod_assignments.first.sample)
            self.dimod_assignments = dimod.SampleSet.from_samples_cqm(cqm_sample, self.model.cqm)

        elif self.model.type_str == "DQM":
            try:
                cqm_sample = self.model.invert(self.dimod_assignments.first.sample)
            except OverflowError:
                cqm_sample = _cqm_invert_safe(self.model.invert, self.dimod_assignments.first.sample)
            dqm_sample = {}
            for s, c in cqm_sample:
                if cqm_sample[(s, c)] == 1:
                    dqm_sample[s] = c
            self.dimod_assignments = dimod.SampleSet.from_samples(dimod.as_samples(dqm_sample), "DISCRETE", 0)

        return self.dimod_assignments
