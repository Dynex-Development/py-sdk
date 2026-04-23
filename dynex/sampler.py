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
"""

import ast
import json
import logging
import math
import multiprocessing
import os
import queue
import secrets
import subprocess
import threading
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

from dynex._solution_parser import (
    SolutionMetrics,
    coerce_float,
    coerce_int,
    decode_varint,
    decompress_bytes,
    extract_solution_stats,
    parse_solution_numbers,
    parse_solution_subject,
    protobuf_has_field,
    sanitize_solution_name,
    skip_field,
    skip_group,
    solution_metrics_from_filename,
)
from dynex._voltage import ensure_voltage_text, extract_voltage_values, process_voltage_line
from dynex.api import DynexAPI
from dynex.config import DynexConfig
from dynex.exceptions import DynexJobError, DynexModelError, DynexValidationError
from dynex.models import BQM
from dynex.proto import sdk_pb2

try:
    import zstandard as zstd
except ModuleNotFoundError:
    zstd = None  # type: ignore


@dataclass
class SamplingTiming:
    start: float = field(default_factory=time.time)
    job_created: float | None = None
    first_shot: float | None = None
    all_shots: float | None = None
    end: float | None = None


def to_wcnf_string(clauses, num_variables, num_clauses):
    """Convert clauses to WCNF string format."""

    line = "p wcnf %d %d\n" % (num_variables, num_clauses)
    for clause in clauses:
        line += " ".join(str(int(lit)) for lit in clause) + " 0\n"
    return line


class DynexSampler:
    """Dynex neuromorphic computing sampler."""

    def __init__(
        self,
        model,
        logging=True,
        description: Optional[str] = None,
        test=False,
        bnb=True,
        filename_override="",
        config: DynexConfig = None,
        preserve_solutions: Optional[bool] = None,
        job_metadata: Optional[dict] = None,
    ):

        # multi-model parallel sampling

        if not config:
            config = DynexConfig()

        self.config = config
        self.logger = config.logger
        self.state = "initialised"
        self.model = model
        self.logging = logging
        self.filename_override = filename_override
        self.description = description if description is not None else config.default_description
        self.test = test
        self.dimod_assignments = {}
        self.bnb = bnb
        self.preserve_solutions = preserve_solutions if preserve_solutions is not None else config.preserve_solutions
        self.use_notebook_output = config.use_notebook_output
        self.job_metadata = job_metadata
        self.timeout = config.default_timeout

    def _log_debug(self, message: str) -> None:
        if self.logger and self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(message)

    @staticmethod
    def _sample_thread(
        q,
        x,
        model,
        logging,
        logger,
        description,
        num_reads,
        annealing_time,
        switchfraction,
        alpha,
        beta,
        gamma,
        delta,
        epsilon,
        zeta,
        minimum_stepsize,
        block_fee,
        shots,
        preserve_solutions,
        qpu_max_coeff,
        config,
    ):
        """Creates a thread for clone sampling."""
        if logging:
            logger.info(f"Clone {x} started...")
        _sampler = _DynexSampler(
            model,
            logging=False,
            description=description,
            test=False,
            config=config,
            preserve_solutions=preserve_solutions,
        )
        _sampleset = _sampler.sample(
            num_reads,
            annealing_time,
            switchfraction,
            alpha,
            beta,
            gamma,
            delta,
            epsilon,
            zeta,
            minimum_stepsize,
            False,
            block_fee,
            shots,
            qpu_max_coeff=qpu_max_coeff,
        )
        if logging:
            logger.info(f"Clone {x} finished")
        q.put(_sampleset)
        return

    def sample(
        self,
        num_reads=32,
        annealing_time=10,
        clones=1,
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
        """Run sampling and return dimod SampleSet."""

        # assert parameters:
        if clones < 1:
            raise DynexValidationError("Value of clones must be in range [1,128]")
        if clones > 128:
            raise DynexValidationError("Value of clones must be in range [1,128]")
        if not self.config.mainnet and clones > 1:
            raise DynexValidationError("Clone sampling is only supported in network mode")

        # Apollo QPU limitation: annealing_time cannot exceed 10000
        MAX_ANNEALING_TIME_QPU = 10000
        compute_backend = getattr(self.config, "compute_backend", None)
        if compute_backend and (compute_backend == "qpu" or compute_backend == "QPU"):
            if annealing_time > MAX_ANNEALING_TIME_QPU:
                if self.logging:
                    self.logger.warning(
                        f"annealing_time ({annealing_time}) exceeds Apollo QPU limit ({MAX_ANNEALING_TIME_QPU}), "
                        f"capping to {MAX_ANNEALING_TIME_QPU}"
                    )
                annealing_time = MAX_ANNEALING_TIME_QPU

        if clones == 1:
            _sampler = _DynexSampler(
                self.model,
                logging=self.logging,
                description=self.description,
                test=self.test,
                bnb=self.bnb,
                filename_override=self.filename_override,
                config=self.config,
                preserve_solutions=self.preserve_solutions,
                job_metadata=self.job_metadata,
            )
            _sampleset = _sampler.sample(
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
            return _sampleset

        else:
            supported_threads = multiprocessing.cpu_count()
            if clones > supported_threads:
                self.logger.info(
                    f"Number of clones > CPU cores: clones: {clones} threads available: {supported_threads}"
                )
            jobs = []
            results = []

            if self.logging:
                self.logger.info(f"Starting sampling ({clones} clones)...")

            # define n samplers:
            for i in range(clones):
                q = multiprocessing.Queue()
                results.append(q)
                p = multiprocessing.Process(
                    target=self._sample_thread,
                    args=(
                        q,
                        i,
                        self.model,
                        self.logging,
                        self.logger,
                        self.description,
                        num_reads,
                        annealing_time,
                        switchfraction,
                        alpha,
                        beta,
                        gamma,
                        delta,
                        epsilon,
                        zeta,
                        minimum_stepsize,
                        block_fee,
                        shots,
                        self.preserve_solutions,
                        qpu_max_coeff,
                        self.config,
                    ),
                )
                jobs.append(p)
                p.start()

            # wait for samplers to finish:
            for job in jobs:
                job.join()

            # collect samples for each job:
            assignments_cum = []
            for result in results:
                assignments = result.get()
                assignments_cum.append(assignments)

            # accumulate and aggregate all results:
            r = None
            for assignment in assignments_cum:
                if len(assignment) > 0:
                    if r is None:
                        r = assignment
                    else:
                        r = dimod.concatenate((r, assignment))

            # aggregate samples:
            r = r.aggregate()

            self.dimod_assignments = r

            return r


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


class _DynexSampler:
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
        config: DynexConfig = None,
        preserve_solutions: Optional[bool] = None,
        job_metadata: Optional[dict] = None,
    ):

        if model.type not in ("cnf", "wcnf", "qasm"):
            raise DynexModelError(f"Unsupported model type: {model.type}")

        self.config = config if config is not None else DynexConfig()
        self.description = description if description is not None else self.config.default_description
        self.api = DynexAPI(config=self.config, logging=logging)
        self.logger = self.config.logger
        self.job_metadata = job_metadata  # Store job_metadata for Circuit vs Constraint BQM distinction
        # FTP removed - using gRPC only

        # local path where tmp files are stored
        # tmppath = Path("tmp/test.bin")
        # tmppath.parent.mkdir(exist_ok=True)
        # with open(tmppath, 'w') as f:
        #     f.write('0123456789ABCDEF')
        self.filepath = "tmp/"
        self.filepath_full = os.getcwd() + "/tmp"
        self.current_job_id = None
        self._downloaded_solutions = set()
        self._grpc_solution_queue = queue.Queue()
        self._grpc_stream_thread = None
        self._grpc_stream_stop = threading.Event()
        self._grpc_stream_lock = threading.Lock()
        self._grpc_active_call = None
        self._grpc_last_seq = 0
        self._grpc_subscription_disabled = False
        self._grpc_subscription_bootstrap_done = False
        self._grpc_solution_meta = {}
        self._job_error = None
        self._grpc_solution_stats = {}
        self._grpc_solution_remote = {}
        self._grpc_downloaded_files = set()
        self._solution_cache: dict = {}
        self.preserve_solutions = (
            preserve_solutions if preserve_solutions is not None else self.config.preserve_solutions
        )
        self.use_notebook_output = self.config.use_notebook_output
        self.timeout = self.config.default_timeout

        # path to testnet
        self.solver_path = self.config.solver_path
        self.bnb = bnb

        # multi-model parallel sampling?
        multi_model_mode = False
        if isinstance(model, list):
            if not self.config.mainnet:
                raise DynexValidationError("Multi model parallel sampling is only supported in network mode")
            multi_model_mode = True

        self.multi_model_mode = multi_model_mode

        # single model sampling:
        if not multi_model_mode:
            # auto generated temp filename:
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

        # multi model sampling:
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

    def _log_debug(self, message: str) -> None:
        if self.logger and self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(message)

    @staticmethod
    def _check_list_length(lst: list) -> bool:
        """Returns True if the sat problem is k-Sat, False for 2-sat or 3-sat."""
        for sublist in lst:
            if isinstance(sublist, list) and len(sublist) > 3:
                return True
        return False

    @staticmethod
    def _save_cnf(clauses: list, filename: str) -> None:
        """Save model as CNF file locally."""
        num_variables = max(max(abs(lit) for lit in clause) for clause in clauses)
        num_clauses = len(clauses)

        with open(filename, "w") as f:
            line = "p cnf %d %d" % (num_variables, num_clauses)

            line_enc = line
            f.write(line_enc + "\n")

            for clause in clauses:
                line = " ".join(str(int(lit)) for lit in clause) + " 0"
                line_enc = line
                f.write(line_enc + "\n")

    def _save_wcnf(self, clauses, filename, num_variables, num_clauses, var_mappings):
        """Save model as WCNF file locally."""
        with open(filename, "w") as f:
            line = "p qubo %d %d %f" % (num_variables, num_clauses, clauses[1])
            f.write(line + "\n")
            for (i, j), value in clauses[0].items():
                if var_mappings:
                    i = next((k for k, v in var_mappings.items() if v == i), i)  # i if not mapped
                    j = next((k for k, v in var_mappings.items() if v == j), j)  # j if not mapped
                line = "%d %d %f" % (i, j, value)
                f.write(line + "\n")

    def _energy(self, sample, mapping=True):
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
                # 2-lit clause:
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
                # 3-lit clause:
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

    def add_salt_local(self):
        """Add salt to local solution filenames for uniqueness."""
        directory = self.filepath_full
        fn = self.filename + "."

        # search for current solution files:
        for filename in os.listdir(directory):
            if filename.startswith(fn):
                # check if salt already added:
                if filename.split(".")[-1].isnumeric():
                    os.rename(directory + "/" + filename, directory + "/" + filename + "." + secrets.token_hex(16))
        return

    def list_files_with_text_local(self):
        """List available solution keys (in-memory for mainnet, files for local solver)."""
        if self.config.mainnet:
            return list(self._solution_cache.keys())

        directory = self.filepath_full
        fn = self.filename + "."
        filtered_files = []

        for filename in os.listdir(directory):
            if filename.startswith(fn) and not filename.endswith("model"):
                if os.path.getsize(os.path.join(directory, filename)) > 0:
                    filtered_files.append(filename)

        return filtered_files

    def validate_file(self, file, debugging=False):
        return True

    def _try_cancel_job(self, job_id, mainnet: bool) -> None:
        if not mainnet or job_id is None:
            return
        try:
            self.api.cancel_job_api(job_id)
        except Exception:
            pass

    def _stop_grpc_subscription(self):
        thread = self._grpc_stream_thread
        if thread is None:
            return

        self._grpc_stream_stop.set()
        with self._grpc_stream_lock:
            call = self._grpc_active_call
        if call is not None and hasattr(call, "cancel"):
            try:
                call.cancel()
            except Exception:
                pass

        thread.join(timeout=5)
        if thread.is_alive() and self.logging:
            self.logger.warning("gRPC subscription thread did not stop in time")

        self._grpc_stream_thread = None
        self._grpc_active_call = None
        self._grpc_stream_stop.clear()

    def _reset_grpc_subscription(self):
        self._stop_grpc_subscription()
        self._grpc_solution_queue = queue.Queue()
        self._grpc_last_seq = 0
        self._grpc_subscription_disabled = False
        self._grpc_subscription_bootstrap_done = False
        self._grpc_solution_meta = {}
        self._grpc_solution_stats = {}
        self._job_error = None
        self._grpc_solution_remote = {}
        self._grpc_downloaded_files = set()
        self._solution_cache = {}

    def _ensure_grpc_subscription(self):
        if self._grpc_subscription_disabled:
            return
        if self.current_job_id is None:
            return
        if self._grpc_stream_thread is not None and self._grpc_stream_thread.is_alive():
            return
        if grpc is None:  # pragma: no cover - guarded by transport selection
            raise ModuleNotFoundError("grpc is required for gRPC communication mode")

        self._grpc_stream_stop.clear()
        self._log_debug(
            f"Starting gRPC solution subscription thread job_id={self.current_job_id} from_seq={self._grpc_last_seq}"
        )
        self._grpc_stream_thread = threading.Thread(
            target=self._grpc_solution_worker,
            name="dynex-grpc-solution",
            daemon=True,
        )
        self._grpc_stream_thread.start()

    def _grpc_solution_worker(self):
        backoff = 1.0
        while not self._grpc_stream_stop.is_set():
            job_id = self.current_job_id
            if job_id is None:
                time.sleep(0.1)
                continue

            try:
                self._log_debug(f"Subscribing to job events job_id={job_id} from_seq={self._grpc_last_seq}")
                call = self.api.subscribe_job_events(job_id, self._grpc_last_seq)
            except NotImplementedError:
                self._grpc_subscription_disabled = True
                self._log_debug("SubscribeJob RPC not implemented; disabling subscription thread")
                return
            except Exception as exc:  # pragma: no cover - network interaction
                if self.logging:
                    self.logger.warning("Failed to connect to server, retrying...")
                self._log_debug(f"SubscribeJob exception job_id={job_id}: {exc}")
                time.sleep(min(backoff, 5.0))
                backoff = min(backoff * 2, 5.0)
                continue

            backoff = 1.0
            with self._grpc_stream_lock:
                self._grpc_active_call = call

            try:
                for event in call:
                    if self._grpc_stream_stop.is_set() or job_id != self.current_job_id:
                        break
                    try:
                        self._grpc_last_seq = event.seq
                    except AttributeError:
                        continue
                    if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                        seq = getattr(event, "seq", "n/a")
                        try:
                            type_name = sdk_pb2.JobEvent.Type.Name(event.type)
                        except ValueError:
                            type_name = str(event.type)
                        has_solution = self._protobuf_has_field(event, "solution")
                        has_envelope = self._protobuf_has_field(event, "envelope")
                        self.logger.debug(
                            "Job event received job_id=%s seq=%s type=%s has_solution=%s has_envelope=%s"
                            % (job_id, seq, type_name, has_solution, has_envelope)
                        )
                    if event.type == sdk_pb2.JobEvent.Type.ERROR:
                        # Handle error events (e.g., Apollo limit exceeded, Modal failures)
                        if self._protobuf_has_field(event, "error"):
                            error_info = event.error
                            error_code = getattr(error_info, "code", "UNKNOWN")
                            error_message = getattr(error_info, "message", "Job failed")
                            self._log_debug(
                                f"Received error event job_id={job_id} seq={event.seq} code={error_code} message={error_message}"
                            )
                            if self.logging:
                                self.logger.error(f"Job {job_id} failed: {error_message}")
                            # Store error and stop waiting
                            self._job_error = error_message
                            self._grpc_stream_stop.set()
                            break
                    elif event.type == sdk_pb2.JobEvent.Type.SOLUTION_NEW:
                        solution_payload = None
                        if self._protobuf_has_field(event, "envelope"):
                            solution_payload = event.envelope
                        elif self._protobuf_has_field(event, "solution"):
                            solution_payload = event.solution
                        if solution_payload is not None:
                            solution_name = getattr(solution_payload, "name", "") or getattr(
                                solution_payload, "checksum", ""
                            )
                            # Log solution details for debugging
                            kind = getattr(solution_payload, "kind", "")
                            has_data = hasattr(solution_payload, "data") and bool(getattr(solution_payload, "data", ""))
                            has_url = bool(getattr(solution_payload, "url", ""))
                            data_len = len(getattr(solution_payload, "data", "")) if has_data else 0
                            self._log_debug(
                                f"Queueing SOLUTION_NEW job_id={job_id} seq={event.seq} name={solution_name} kind={kind} has_data={has_data} has_url={has_url} data_len={data_len}"
                            )
                            if self.logging:
                                self.logger.debug(
                                    f"Received solution event job_id={job_id} name={solution_name} kind={kind} has_inline={has_data} has_url={has_url} data_len={data_len}"
                                )
                            self._grpc_solution_queue.put(solution_payload)
                        else:
                            self._log_debug(
                                f"SOLUTION_NEW event has no envelope or solution payload job_id={job_id} seq={event.seq}"
                            )
                            if self.logging:
                                self.logger.warning(
                                    f"SOLUTION_NEW event missing payload job_id={job_id} seq={event.seq}"
                                )
            except grpc.RpcError as exc:  # pragma: no cover - network interaction
                code = exc.code() if hasattr(exc, "code") else None
                if self._grpc_stream_stop.is_set() and code == grpc.StatusCode.CANCELLED:
                    return
                if code == grpc.StatusCode.UNIMPLEMENTED:
                    self._grpc_subscription_disabled = True
                    if self.logging:
                        self.logger.info("SubscribeJob RPC unavailable, falling back to legacy polling")
                    self._log_debug("SubscribeJob RPC returned UNIMPLEMENTED; enabling polling fallback")
                    return
                if self.logging and code != grpc.StatusCode.CANCELLED:
                    self.logger.warning("Connection to server interrupted, reconnecting...")
                self._log_debug(f"SubscribeJob stream interrupted job_id={job_id} code={code} error={exc}")
                time.sleep(1.0)
                continue
            except Exception as exc:  # pragma: no cover - network interaction
                if self.logging:
                    self.logger.warning("Lost connection to server, retrying...")
                self._log_debug(f"SubscribeJob stream error job_id={job_id}: {exc}")
                time.sleep(1.0)
                continue
            finally:
                with self._grpc_stream_lock:
                    self._grpc_active_call = None

            if self._grpc_stream_stop.is_set():
                return
            time.sleep(0.1)

    def _consume_solution_meta(self, solution) -> None:
        if self.logger and self.logger.isEnabledFor(logging.DEBUG):
            try:
                fields = {desc.name: value for desc, value in solution.ListFields()}  # type: ignore[attr-defined]
            except AttributeError:
                fields = {}
            else:
                # Avoid logging potentially huge payloads inline
                for key in list(fields):
                    value = fields[key]
                    if isinstance(value, (bytes, bytearray)) and len(value) > 128:
                        fields[key] = f"<bytes {len(value)}>"
            self._log_debug(f"Solution meta fields: {fields}")
            unknown = getattr(solution, "_unknown_fields", None)
            if unknown:
                self._log_debug(f"Solution meta unknown fields: {unknown}")
        remote_name = getattr(solution, "name", "") or getattr(solution, "checksum", "")
        checksum = getattr(solution, "checksum", "")
        size_hint = getattr(solution, "size", None)
        valid_flag = getattr(solution, "valid", None)

        # If remote_name looks like solution data (contains commas and numbers), use checksum or generate a hash
        # Backend sometimes sends solution data as name instead of a proper filename
        if remote_name and ("," in remote_name or len(remote_name) > 200):
            # This is likely solution data, not a filename - use checksum or generate hash
            if checksum:
                safe_name = self._sanitize_solution_name(checksum)
                if self.logging:
                    self.logger.warning(
                        f"remote_name appears to be solution data (length={len(remote_name)}), "
                        f"using checksum instead: {checksum[:20]}..."
                    )
            else:
                # Generate a hash from the name/data as fallback
                import hashlib

                name_hash = hashlib.md5(
                    remote_name.encode("utf-8") if isinstance(remote_name, str) else remote_name
                ).hexdigest()
                safe_name = f"solution_{name_hash[:16]}"
                if self.logging:
                    self.logger.warning(
                        f"remote_name appears to be solution data (length={len(remote_name)}), "
                        f"generated hash-based name: {safe_name}"
                    )
        else:
            safe_name = self._sanitize_solution_name(remote_name or checksum)
        self._grpc_solution_remote[safe_name] = remote_name
        meta_stats = self._extract_solution_stats(solution, remote_name)
        stats_copy = dict(meta_stats) if meta_stats else {}
        compression_hint = (
            getattr(solution, "compression", None) or stats_copy.get("compression") or stats_copy.get("Compression")
        )
        if compression_hint and "compression" not in stats_copy:
            stats_copy["compression"] = compression_hint
        compressed_size_hint = getattr(solution, "compressed_size", 0)
        if compressed_size_hint and "compressed_size" not in stats_copy:
            stats_copy["compressed_size"] = compressed_size_hint
        for key in ("chips", "steps", "loc"):
            if key in stats_copy:
                stats_copy[key] = self._coerce_int(stats_copy[key])
        if "energy" in stats_copy:
            stats_copy["energy"] = self._coerce_float(stats_copy["energy"])
        self._grpc_solution_meta[safe_name] = solution
        if stats_copy:
            preview = {k: stats_copy[k] for k in stats_copy if k != "subject_dict"}
            self._log_debug(f"Solution stats extracted name={remote_name} keys={list(preview.keys())}")
        else:
            self._log_debug(f"No stats extracted for solution name={remote_name}")
        kind = getattr(solution, "kind", "")
        url = getattr(solution, "url", "")
        # Check if data field exists and get it
        inline_data = ""
        if hasattr(solution, "data"):
            inline_data = getattr(solution, "data", "") or ""
        subject = getattr(solution, "subject", "")
        self._log_debug(
            f"Processing solution meta job_id={self.current_job_id} name={remote_name} "
            f"checksum={checksum} size={size_hint} valid={valid_flag} kind={kind} url={bool(url)} inline={bool(inline_data)} safe={safe_name} subject={subject!r}"
        )
        if subject:
            self._log_debug(f"gRPC solution meta subject name={remote_name} kind={kind} length={len(subject)}")
        if not remote_name:
            self._log_debug("Solution meta skipped: missing remote name")
            return
        if hasattr(solution, "valid") and not solution.valid:
            self._log_debug(f"Solution meta skipped: marked invalid name={remote_name}")
            return
        if remote_name in self._downloaded_solutions:
            self._log_debug(f"Solution meta skipped: already downloaded name={remote_name}")
            return
        if self.current_job_id is None:
            self._log_debug(f"Solution meta skipped: job not active name={remote_name}")
            return

        # Ensure local_name is not too long (filesystem limit is typically 255 chars)
        # If safe_name is still too long, truncate it
        max_name_length = 200  # Leave room for filename prefix and extension
        if len(safe_name) > max_name_length:
            # Truncate and add hash suffix to ensure uniqueness
            import hashlib

            name_hash = hashlib.md5(safe_name.encode("utf-8")).hexdigest()[:8]
            safe_name = safe_name[: max_name_length - 9] + "_" + name_hash
            if self.logging:
                self.logger.warning(f"safe_name too long ({len(safe_name)}), truncated to: {safe_name[:50]}...")

        local_name = f"{self.filename}.{safe_name}"
        local_path = os.path.join(self.filepath, local_name)

        # Final check: if path is still too long, use a hash-based name
        if len(local_path) > 250:  # Conservative limit
            import hashlib

            path_hash = hashlib.md5(local_path.encode("utf-8")).hexdigest()[:16]
            local_name = f"{self.filename}.sol_{path_hash}"
            local_path = os.path.join(self.filepath, local_name)
            if self.logging:
                self.logger.warning(f"Local path too long, using hash-based name: {local_name}")

        # Check if already received (cache for mainnet, file for local solver)
        if self.config.mainnet:
            if local_name in self._solution_cache:
                self._downloaded_solutions.add(remote_name)
                self._log_debug(f"Solution already in cache name={remote_name} key={local_name}")
                return
        else:
            if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
                self._downloaded_solutions.add(remote_name)
                self._log_debug(f"Solution already present locally name={remote_name} path={local_path}")
                return

        raw_data: bytes | None = None

        # Decode inline data (kind == "inline")
        if inline_data and kind == "inline":
            try:
                self._log_debug(
                    f"Processing inline solution data name={remote_name} kind={kind} data_len={len(inline_data)}"
                )
                import base64

                try:
                    compressed_data = base64.b64decode(inline_data)
                except Exception as decode_exc:
                    if self.logging:
                        self.logger.warning(f"Base64 decode failed, trying raw data: {decode_exc}")
                    compressed_data = inline_data.encode("utf-8") if isinstance(inline_data, str) else inline_data

                raw_data = (
                    self._decompress_bytes(compressed_data, compression_hint) if compression_hint else compressed_data
                )
                self._log_debug(f"Inline solution decoded name={remote_name} size={len(raw_data)}")
            except Exception as exc:
                if self.logging:
                    self.logger.error(f"Inline data processing error {remote_name}: {exc}", exc_info=True)
                self._log_debug(f"Inline data processing failed name={remote_name} error={exc}")

        # Fetch via presigned URL if inline data was not available
        if raw_data is None and url:
            try:
                self._log_debug(f"Downloading solution via presigned URL name={remote_name} url={url[:50]}...")
                import urllib.request

                with urllib.request.urlopen(url) as response:
                    fetched = response.read()
                raw_data = self._decompress_bytes(fetched, compression_hint) if compression_hint else fetched
            except Exception as exc:
                if self.logging:
                    self.logger.error(f"Presigned URL download error {remote_name}: {exc}")
                self._log_debug(f"Presigned URL download failed name={remote_name} error={exc}")

        if raw_data is None:
            self._log_debug(
                f"Solution not available: no URL and no inline data name={remote_name} kind={kind} "
                f"has_inline={bool(inline_data)} has_url={bool(url)}"
            )
            return

        # Store solution: in-memory for mainnet, on disk for local solver.
        # Write to disk only when debug_save_solutions is enabled (debug/analysis).
        if self.config.mainnet:
            self._solution_cache[local_name] = raw_data
            if self.config.debug_save_solutions:
                try:
                    os.makedirs(os.path.dirname(local_path) or self.filepath, exist_ok=True)
                    with open(local_path, "wb") as fh:
                        fh.write(raw_data)
                    self._log_debug(f"Debug: solution saved to disk name={remote_name} path={local_path}")
                except OSError as exc:
                    if self.logging:
                        self.logger.warning(f"Debug file write failed {local_path}: {exc}")
        else:
            os.makedirs(os.path.dirname(local_path) or self.filepath, exist_ok=True)
            with open(local_path, "wb") as fh:
                fh.write(raw_data)
            self._log_debug(f"Solution saved to disk name={remote_name} path={local_path} size={len(raw_data)}")

        # Validate (currently always returns True; kept for future checks)
        if not self.validate_file(local_name):
            self._log_debug(f"Solution validation failed name={remote_name}")
            if self.logging:
                self.logger.warning(f"Solution validation failed for {remote_name}, discarding")
            if self.config.mainnet:
                self._solution_cache.pop(local_name, None)
            elif os.path.exists(local_path):
                os.remove(local_path)
            try:
                self.api.report_invalid(filename=remote_name, reason="wrong energy reported")
            except Exception:  # pragma: no cover - best effort
                pass
            return

        self._downloaded_solutions.add(remote_name)
        solutions_count = len(self._downloaded_solutions)

        if self.logging:
            self.logger.debug(f"Solution added to downloaded_solutions: {remote_name}, total_count={solutions_count}")
            expected = getattr(self, "expected_shots", None)
            if expected:
                self.logger.info(f"Shot {solutions_count}/{expected} received")
                timing = getattr(self, "_timing", None)
                if timing and timing.first_shot is None and solutions_count == 1:
                    timing.first_shot = time.time()

        local_basename = os.path.basename(local_path)
        self._grpc_downloaded_files.add(local_name)
        key_candidates = {safe_name, local_name, local_basename, remote_name}
        for key in list(key_candidates):
            if not key:
                continue
            self._grpc_solution_remote[key] = remote_name
            if stats_copy:
                self._grpc_solution_stats[key] = dict(stats_copy)
        if stats_copy:
            self._grpc_solution_stats[local_name] = dict(stats_copy)
        self._grpc_solution_meta[local_name] = solution
        self._log_debug(
            f"Solution stored name={remote_name} key={local_name} downloaded_count={len(self._downloaded_solutions)}"
        )

    def list_files_with_text(self):
        """Fetch assignment files from gRPC and return list of local paths."""
        if self.config.mainnet:
            self._list_files_with_text_grpc()
            return

        # FTP removed - using gRPC only

    _sanitize_solution_name = staticmethod(sanitize_solution_name)
    _parse_solution_subject = staticmethod(parse_solution_subject)
    _parse_solution_numbers = staticmethod(parse_solution_numbers)
    _coerce_int = staticmethod(coerce_int)
    _coerce_float = staticmethod(coerce_float)
    _decompress_bytes = staticmethod(decompress_bytes)
    _decode_varint = staticmethod(decode_varint)
    _skip_field = staticmethod(skip_field)  # type: ignore[assignment]
    _skip_group = staticmethod(skip_group)  # type: ignore[assignment]
    _protobuf_has_field = staticmethod(protobuf_has_field)

    def _extract_solution_stats(self, solution, remote_name: str) -> dict:
        return extract_solution_stats(solution, remote_name)

    def _lookup_grpc_stats(self, filename: str, info: str) -> dict:
        stats = self._grpc_solution_stats.get(filename, {})
        if not stats:
            stats = self._grpc_solution_stats.get(info, {})
        if not stats:
            remote_lookup = self._grpc_solution_remote.get(filename) or self._grpc_solution_remote.get(info)
            if remote_lookup:
                stats = self._grpc_solution_stats.get(remote_lookup, {})
        return stats

    def _solution_metrics_from_filename(self, filename: str, fallback_info: str, stats: dict) -> SolutionMetrics:
        return solution_metrics_from_filename(filename, fallback_info, stats)

    def _get_solution_metrics(self, filename: str) -> SolutionMetrics:
        info = filename[len(self.filename) + 1 :]
        stats = {}
        if self.config.mainnet:
            stats = self._lookup_grpc_stats(filename, info)
        return solution_metrics_from_filename(filename, info, stats)

    def _list_files_with_text_ftp(self):
        """FTP method removed - using gRPC only"""
        pass

    def _list_files_with_text_grpc(self):
        if self.current_job_id is None:
            return

        if self._grpc_subscription_disabled:
            # Streaming is required - polling via ListSolutions is no longer supported
            raise NotImplementedError(
                "gRPC subscription (SubscribeJob) is required for solution delivery. "
                "ListSolutions polling has been removed. Please ensure SubscribeJob RPC is available."
            )

        try:
            self._ensure_grpc_subscription()
        except NotImplementedError:
            raise
        except Exception as exc:
            if self.logging:
                self.logger.error("Failed to establish server connection for solution delivery")
            self._log_debug(f"gRPC subscription error: {exc}")
            raise

        drained = False
        drained_count = 0
        while True:
            try:
                solution = self._grpc_solution_queue.get_nowait()
            except queue.Empty:
                break
            drained = True
            drained_count += 1
            try:
                if self.logging:
                    solution_name = getattr(solution, "name", "") or getattr(solution, "checksum", "")
                    kind = getattr(solution, "kind", "")
                    self.logger.debug(
                        f"Processing queued solution: name={solution_name}, kind={kind}, "
                        f"queue_size={self._grpc_solution_queue.qsize()}"
                    )
                self._consume_solution_meta(solution)
            except Exception as exc:
                if self.logging:
                    self.logger.error(f"Error processing queued solution: {exc}", exc_info=True)
                self._log_debug(f"Error processing queued solution: {exc}")
            finally:
                self._grpc_solution_queue.task_done()

        if drained and drained_count > 0:
            self._log_debug(f"Processed {drained_count} queued solutions job_id={self.current_job_id}")
            if self.logging:
                self.logger.debug(
                    f"Processed {drained_count} queued solutions job_id={self.current_job_id}, "
                    f"downloaded_count={len(self._downloaded_solutions)}"
                )

        # Bootstrap is no longer needed - streaming will deliver all solutions
        if not drained:
            self._grpc_subscription_bootstrap_done = True

    def _clean(self):
        """Cleanup after sampling - removes unused solution files."""
        if self.config.mainnet:
            self.list_files_with_text_local()

    def __exit__(self, exc_type, exc_value, traceback):
        """Cleanup on sampler exit."""
        self.logger.info("Sampler exit")

    def _update(self, model, logging=True):
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
                    self.clauses, self.filepath + self.filename, self.num_variables, self.num_clauses, self.var_mappings
                )

        self.type = model.type
        self.assignments = {}
        self.dimod_assignments = {}
        self.bqm = model.bqm

    def delete_local_files_by_prefix(self, directory: str, prefix: str):
        if self.config.mainnet:
            keys = [k for k in self._solution_cache if k.startswith(prefix)]
            for k in keys:
                del self._solution_cache[k]
            if keys:
                self._log_debug(f"Cleared {len(keys)} cached solutions prefix={prefix}")
            if self.config.debug_save_solutions:
                self._delete_files_by_prefix(directory, prefix)
            return

        self._delete_files_by_prefix(directory, prefix)

    def _delete_files_by_prefix(self, directory: str, prefix: str):
        try:
            entries = os.listdir(directory)
        except OSError:
            return
        for filename in entries:
            if filename.startswith(prefix):
                file_path = os.path.join(directory, filename)
                try:
                    os.remove(file_path)
                    self.logger.info(f"Solution deleted: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to delete solution {file_path}: {e}")

    @staticmethod
    def _convert(a):
        """Convert flat list to dict (key, value pairs)."""
        it = iter(a)
        res_dct = dict(zip(it, it))
        return res_dct

    def _print(self):
        """Print sampler summary (network mode, filename, model type, variables, clauses)."""
        self.logger.info("{DynexSampler object}")
        self.logger.info(f"network_mode? {self.config.mainnet}")
        self.logger.info(f"logging? {self.logging}")
        self.logger.info(f"tmp filename: {self.filepath + self.filename}")
        self.logger.info(f"model type: {self.type_str}")
        self.logger.info(f"num variables: {self.num_variables}")
        self.logger.info(f"num clauses: {self.num_clauses}")
        self.logger.info("configuration loaded")

    def _sample_to_assignments(self, lowest_set):
        """Convert voltage list (-1/+1) to binary sample dict (0/1)."""
        sample = {}
        i = 0
        for var in self.var_mappings:
            sample[var] = 1
            if float(lowest_set[i]) < 0:
                sample[var] = 0
            i = i + 1
        return sample

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
    ):
        """Main sampling entry point - delegates to _sample."""
        retval = {}

        # In a malleable environment, it is rarely possible that a worker is submitting an inconsistent solution file. If the job
        # is small, we need to re-sample again. This routine samples up to NUM_RETRIES (10) times. If an error occurs, or
        # a keyboard interrupt was triggered, the return value is a dict containing key 'error'

        # Store expected number of solutions for progress tracking
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

                # aggregate sampleset:
            if self.type == "wcnf" and len(retval) > 0 and ("error" in retval) is False:
                retval = retval.aggregate()

        return retval

    def read_voltage_data(self, file, mainnet, rank):
        if mainnet:
            data = self._solution_cache.get(file)
            if data is None:
                self.logger.error(f"Solution not in cache: {file}")
                return ["NaN"]
            skip_first = self.type == "qasm"
            if rank == 1:
                return self._extract_voltage_values(data, prefer_last=False, skip_first=skip_first)
            return self._extract_voltage_values(data, prefer_last=(rank > 1), skip_first=skip_first)

        file_path = os.path.join(self.filepath, file)
        try:
            with open(file_path, "rb") as ffile:
                prefer_last = rank > 1
                return self._read_last_non_empty_line(ffile) if prefer_last else self._read_entire_file(ffile)
        except (IOError, OSError) as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return ["NaN"]

    def _read_last_non_empty_line(self, file_obj):
        data = file_obj.read()
        # For QASM, skip first line (energy metrics)
        skip_first = self.type == "qasm"
        return self._extract_voltage_values(data, prefer_last=True, skip_first=skip_first)

    def _read_second_line(self, file_obj):
        data = file_obj.read()
        # For QASM, skip first line (energy metrics)
        skip_first = self.type == "qasm"
        return self._extract_voltage_values(data, prefer_last=False, skip_first=skip_first)

    def _read_entire_file(self, file_obj):
        data = file_obj.read()
        # For QASM, skip first line (energy metrics)
        skip_first = self.type == "qasm"
        return self._extract_voltage_values(data, prefer_last=False, skip_first=skip_first)

    _extract_voltage_values = staticmethod(extract_voltage_values)
    _process_voltage_line = staticmethod(process_voltage_line)
    _ensure_voltage_text = staticmethod(ensure_voltage_text)

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
        # Initialize timing tracking
        self._timing = SamplingTiming()

        if self.multi_model_mode is True:
            raise DynexJobError("Multi-model parallel sampling is not implemented yet")

        # Apollo QPU limitation: annealing_time cannot exceed 10000
        MAX_ANNEALING_TIME_QPU = 10000
        compute_backend = getattr(self.config, "compute_backend", None)
        is_qpu = compute_backend and (compute_backend == "qpu" or compute_backend == "QPU")

        if is_qpu:
            # Get qpu_model from config (required for QPU backend)
            qpu_model = self.config.qpu_model

            # Store qpu_model in job_metadata for QPU jobs
            if self.job_metadata is None:
                self.job_metadata = {}
            self.job_metadata["qpu_model"] = str(qpu_model)  # Convert enum to string if needed

            if self.logging:
                self.logger.info(f"Apollo QPU chip: {qpu_model}")
                # Settings summary
                # For Circuit BQM (QASM), num_variables/num_clauses are determined after API conversion
                is_circuit_bqm = self.job_metadata and self.job_metadata.get("type") == "qasm"
                if not is_circuit_bqm and self.num_variables is not None:
                    self.logger.info(f"Problem: {self.num_variables} qubits, {self.num_clauses} gates")
                self.logger.info(f"Settings: num_reads={num_reads}, shots={shots}, annealing_time={annealing_time}")

                # Validation warnings
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

            # Automatic BQM scaling for QPU (Constraint BQM only, Circuit BQM handled by Apollo API)
            # Skip scaling for Circuit BQM (QASM) as it's handled automatically by Apollo API
            is_circuit_bqm = self.job_metadata and self.job_metadata.get("type") == "qasm"

            if self.bqm and not is_circuit_bqm:
                from .preprocessing import scale_bqm_to_range

                # Check if BQM needs scaling
                max_abs = 0.0
                for coeff in self.bqm.linear.values():
                    max_abs = max(max_abs, abs(float(coeff)))
                for coeff in self.bqm.quadratic.values():
                    max_abs = max(max_abs, abs(float(coeff)))

                # Only scale if coefficients exceed user-specified threshold
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

        # ensure correct ground state display:
        if self.bqm and not preprocess:
            self.model.wcnf_offset = self.bqm.offset
            self.model.precision = 1

        # Preprocess (only for non-QPU backends):
        # For QPU backend, preprocess parameter is passed to Apollo and executed there
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
            elapsed_time = end_time - start_time  # in s
            elapsed_time *= 100

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
                # Use job_metadata from __init__ or auto-create for Circuit BQM (QASM)
                job_metadata = self.job_metadata  # First, try from __init__
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
                    "job_metadata": job_metadata,  # Automatically set for Circuit BQM
                }

                job_id, self.filename, price_per_block, qasm = self.api.create_job_api_proto(
                    **params, debugging=debugging
                )
                self._timing.job_created = time.time()
                self._reset_grpc_subscription()
                self.current_job_id = job_id
                self._downloaded_solutions.clear()
                # show effective price in DNX:
                price_per_block = price_per_block / 1000000000
                # parse qasm data:
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
                        # construct circuit model:
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
                    # Update Problem info log for QPU Circuit BQM after QASM conversion
                    if self.logging and is_qpu:
                        self.logger.info(
                            f"Problem: {self.num_variables} qubits, {self.num_clauses} gates (Circuit BQM from QASM)"
                        )
                if self.logging:
                    # Settings summary for non-QPU backends
                    if not is_qpu:
                        self.logger.info(f"Problem: {self.num_variables} qubits, {self.num_clauses} gates")
                        self.logger.info(
                            f"Settings: num_reads={num_reads}, shots={shots}, annealing_time={annealing_time}"
                        )

                        # Validation warnings for non-QPU backends
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
                # run on test-net:
                if self.type == "qasm":
                    # testnet qasm sampling requires a dedicated library (not in default package):
                    command = (
                        "python3 dynex_circuit_backend.py --mainnet False --file "
                        + self.model.qasm_filepath
                        + self.model.qasm_filename
                    )
                    if debugging:
                        command = command + " --debugging True"
                        # Debug mode: show output via logger
                        process = subprocess.Popen(
                            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
                        )
                        if self.logging:
                            for line in iter(process.stdout.readline, ""):
                                if line.strip():
                                    self.logger.debug(f"[QASM] {line.rstrip()}")
                        process.wait()
                    else:
                        # Normal mode: redirect to DEVNULL
                        if self.logging:
                            self.logger.info("Waiting for reads...")
                        process = subprocess.Popen(
                            command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                        )
                        process.wait()
                    # read returned model:
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
                    # construct circuit model:
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

                # use branch-and-bound (testnet) sampler instead?:
                bnb_binary = self.solver_path + "dynex-testnet-bnb"
                if self.bnb and os.path.exists(bnb_binary):
                    command = bnb_binary + " " + self.filepath_full + "/" + self.filename
                else:
                    # Always use v2 format (dynexcore)
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
                    # command += " ode_steps=" + str(annealing_time) #
                    # command += " search_steps=" + str(1000000) #
                    # command += " mutation_rate=10"
                    command += " init_dt=" + str(minimum_stepsize)
                    command += " cpu_threads=4"
                    command += " shots=" + str(rank)
                    # self.logger.info(f'[DYNEX DEBUG] Solver command: {command}')

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
            t_start = time.time()  # Real time for timeout check
            finished = False
            runupdated = False
            cnt_workers = 0
            # Timeout for QPU (Apollo API)
            compute_backend = getattr(self.config, "compute_backend", None)
            # Use self.timeout if set, otherwise default to 300 seconds
            max_wait_time = getattr(self, "timeout", 300.0)

            # initialise display:
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
                # Check for job errors (e.g., Apollo limit exceeded, Modal failures)
                if self._job_error:
                    error_msg = f"Job failed: {self._job_error}"
                    if self.logging:
                        self.logger.error(f"{error_msg}")
                    raise DynexJobError(error_msg)

                # Check timeout
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
                                f"Timeout after {max_wait_time:.0f}s, but received {solutions_count} solution(s). Continuing..."
                            )
                    finished = True
                    break

                total_chips = 0
                total_steps = 0
                lowest_energy = 1.7976931348623158e308
                lowest_loc = 1.7976931348623158e308

                # retrieve solutions via gRPC stream (SubscribeJob)
                # Note: list_files_with_text() manages gRPC subscription
                # and drains the solution queue. All solutions are delivered via streaming.
                if mainnet:
                    try:
                        self.list_files_with_text()
                    except Exception:
                        # Continue - errors are non-fatal, streaming will deliver solutions
                        pass

                files = self.list_files_with_text_local()
                cnt_workers = len(files)

                # Debug: log every 10 seconds to show SDK is alive
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
                        f"Still waiting for solutions... elapsed={elapsed_int}s, timeout={int(max_wait_time)}s, job_id={self.current_job_id}, files={len(files)}, downloaded={len(self._downloaded_solutions)}"
                    )

                # Only log when there are changes
                if debugging and not mainnet and len(files) != prev_files_count:
                    self.logger.debug(
                        f"Local mode: found {len(files)} solution files, cnt_solutions={self.cnt_solutions}, shots={shots}"
                    )

                solutions_count = len(self._downloaded_solutions)
                if mainnet:
                    self.cnt_solutions = solutions_count
                else:
                    self.cnt_solutions += len(files)

                # Check exit condition BEFORE processing files
                # This ensures we check even when files list is empty
                if mainnet:
                    # Complete if we have enough solutions, regardless of chips
                    # (some solutions may have chips=0 but still be valid)
                    if solutions_count >= shots:
                        self._log_debug(
                            f"Exit condition met: solutions_count={solutions_count} >= shots={shots}, finishing..."
                        )
                        # Track when all shots received
                        if self._timing.all_shots is None:
                            self._timing.all_shots = time.time()
                        finished = True
                    else:
                        # Debug logging (every 5 seconds)
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
                    # Local mode: collect required number of solutions
                    if self.cnt_solutions >= shots:
                        finished = True

                # Update previous counts
                prev_files_count = len(files)

                if not files:
                    # If we have enough solutions, exit immediately
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

                # Exit condition already checked above, but check again after processing files
                # in case we need to update based on file metrics
                if finished:
                    break
                details = ""
                display_energy = lowest_energy if math.isfinite(lowest_energy) else 0.0

                # Only show table at key moments (start, finish, major updates)
                table_update_counter += 1
                should_show_table = (
                    (table_update_counter == 1) or finished or (cnt_workers > 0 and table_update_counter % 10 == 0)
                )

                if self.logging and should_show_table and mainnet:
                    if not debugging and self.use_notebook_output:
                        clear_output(wait=True)

                    # Status details removed - not essential for core functionality
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

                # Add small delay to reduce CPU usage and make output more readable
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

            # update final output (display all workers as stopped as well):
            if cnt_workers > 0 and self.logging:
                if mainnet and not debugging and self.use_notebook_output:
                    clear_output(wait=True)
                details = ""
                display_energy = lowest_energy if math.isfinite(lowest_energy) else 0.0
                if mainnet:
                    # Status details removed - not essential for core functionality
                    details = "*** WAITING FOR WORKERS ***"

                elapsed_time = time.process_time() - t
                if mainnet:
                    # Display results table
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

            # Finalize timing
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

                    # Show average time per shot if multiple shots
                    if shots > 1 and compute_time > 0:
                        avg_time_per_shot = compute_time / shots
                        self.logger.info(
                            f"Average time per shot: {avg_time_per_shot:.2f}s ({shots} shots in {compute_time:.2f}s)"
                        )

                    # Show timing breakdown
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

                # voltages received successfully

                # valid result? ignore Nan values and other incorrect data
                if self.type in ["wcnf", "qasm"] and voltages and len(voltages) > 0:
                    sampleset.append(
                        ["sample", voltages, "chips", chips, "steps", steps, "falsified softs", loc, "energy", energy]
                    )
                    if loc < lowest_loc:
                        lowest_loc = loc
                    if energy < lowest_energy:
                        lowest_energy = energy
                        lowest_set = voltages
                    # add voltages to dimod return sampleset:
                    usable_len = min(len(voltages), self.num_variables)
                    if usable_len < self.num_variables:
                        self._log_debug(
                            f"Voltage result shorter than expected solver=2 got={len(voltages)} expected={self.num_variables}"
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

            # Accept solutions if we have at least 90% of variables (allow small discrepancies)
            min_required_vars = int(self.num_variables * 0.9) if self.num_variables else 0
            if (
                (self.type in ["wcnf", "qasm"])
                and len(lowest_set) >= min_required_vars
                and len(lowest_set) <= self.num_variables
            ):
                sample = {}
                i = 0
                for var in self.var_mappings:
                    # _var = self.var_mappings[var]
                    if i < len(lowest_set):
                        sample[var] = 1
                        if float(lowest_set[i]) < 0:
                            sample[var] = 0
                        i = i + 1
                    else:
                        # Pad missing variables with 0
                        sample[var] = 0
                self.assignments = sample

                # generate dimod format sampleset:
                self.dimod_assignments = dimod.SampleSet.from_samples_bqm(dimod_sample, self.bqm)
            elif self.type in ["wcnf", "qasm"]:
                # If no valid solutions or size mismatch, create empty SampleSet
                self.logger.warning(
                    "No valid sampleset returned from sampling. The job may have been cancelled or failed."
                )
                self.logger.warning(
                    f"lowest_set length ({len(lowest_set)}) does not match num_variables ({self.num_variables})"
                )
                # Create empty SampleSet to avoid returning dict
                self.dimod_assignments = dimod.SampleSet.from_samples_bqm([], self.bqm)

            if self.logging:
                self.logger.info(f"Sampleset ready with energy {self.dimod_assignments}")

            sampleset_clean = []
            for sample in sampleset:
                sample_dict = self._convert(sample)
                sampleset_clean.append(sample_dict)

            # Delete local files: always on mainnet unless preserve_solutions=True,
            # or on local solver when remove_local_solutions=True.
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

        # CQM model?
        if self.model.type_str == "CQM":
            try:
                cqm_sample = self.model.invert(self.dimod_assignments.first.sample)
            except OverflowError:
                cqm_sample = _cqm_invert_safe(self.model.invert, self.dimod_assignments.first.sample)
            self.dimod_assignments = dimod.SampleSet.from_samples_cqm(cqm_sample, self.model.cqm)

        # DQM model?
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
