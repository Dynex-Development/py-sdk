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

import zipfile
from typing import TYPE_CHECKING, Iterator, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from dynex.compute_backend import ComputeBackend
from dynex.config import DynexConfig
from dynex.exceptions import DynexValidationError
from dynex.interfaces.api import Job

if TYPE_CHECKING:
    from dynex.proto import sdk_pb2
try:
    from dynex.grpc_client import DynexGrpcClient
except ModuleNotFoundError as exc:
    if exc.name == "grpc":
        DynexGrpcClient = None
    else:
        raise


class DynexAPI:
    """
    Dynex API client for gRPC communication only.
    """

    def __init__(self, config: DynexConfig = None, logging: bool = True):
        self.config = config if config is not None else DynexConfig()
        self.logger = getattr(self.config, "logger", None)
        self.logging = logging
        self._grpc_client: DynexGrpcClient | None = None

    def _get_grpc_client(self) -> DynexGrpcClient:
        if DynexGrpcClient is None:
            raise ModuleNotFoundError("grpc is not installed. Please install 'grpcio' to use gRPC transport.")
        if self._grpc_client is None:
            self._grpc_client = DynexGrpcClient(self.config, self.logger)
        return self._grpc_client

    def subscribe_job_events(self, job_id: int, from_seq: int = 0) -> Iterator["sdk_pb2.JobEvent"]:
        if not self.config.mainnet:
            raise NotImplementedError("Job subscription is only available in network mode")
        return self._get_grpc_client().subscribe_job(job_id, from_seq)

    def update_job_api(self, job_id: int) -> bool:
        """Update an ongoing job via gRPC."""
        if not self.config.mainnet:
            raise NotImplementedError("Job updates are only available in network mode")
        reply = self._get_grpc_client().update_job(job_id)
        return bool(Job(job_id=reply.job_id, min_loc=reply.min_loc, min_energy=reply.min_energy))

    def report_invalid(self, filename: str, reason: str) -> bool:
        """Report invalid solution file."""
        if not self.config.mainnet:
            raise NotImplementedError("Invalid solution reporting is only available in network mode")
        raise NotImplementedError("gRPC method for reporting invalid solutions not yet implemented")

    def cancel_job_api(self, job_id: int) -> bool:
        """Cancel an ongoing job."""
        if not self.config.mainnet:
            raise NotImplementedError("Job cancellation is only available in network mode")
        reply = self._get_grpc_client().cancel_job(job_id)
        return reply.job_id == job_id

    def finish_job_api(self, job_id: int, min_loc: float, min_energy: float) -> bool:
        """Finish an ongoing job."""
        if not self.config.mainnet:
            raise NotImplementedError("Job finishing is only available in network mode")
        reply = self._get_grpc_client().finish_job(job_id, min_loc, min_energy)
        return reply.job_id == job_id

    def download_solution(self, job_id: int, name: str, destination_path: str) -> None:
        """Download solution file via gRPC."""
        if not self.config.mainnet:
            raise NotImplementedError("Solution download is only available in network mode")
        self._get_grpc_client().download_solution(job_id, name, destination_path)

    def create_job_api(
        self,
        sampler: "DynexSampler",  # noqa: F821
        annealing_time: int,
        switchfraction: int,
        num_reads: int,
        alpha: int = 20,
        beta: int = 20,
        gamma: int = 1,
        delta: int = 1,
        epsilon: int = 1,
        zeta: int = 1,
        minimum_stepsize: float = 0.05,
        block_fee: int = 0,
        shots: int = 1,
        rank: int = 1,
        target_energy: float = 0.0,
        job_metadata: Optional[dict] = None,
    ) -> tuple:
        """Create a new job via gRPC."""
        if not self.config.mainnet:
            raise NotImplementedError("Job creation is only available in network mode")

        target_energy_float = float(target_energy) if hasattr(target_energy, "item") else float(target_energy)
        compute_backend = getattr(sampler.config, "compute_backend", "unspecified")

        opts = JobOptions(
            annealing_time=annealing_time,
            switchfraction=switchfraction,
            num_reads=num_reads,
            params=[alpha, beta, gamma, delta, epsilon, zeta],
            min_step_size=minimum_stepsize,
            description=sampler.description,
            block_fee=block_fee,
            shots=shots,
            target_energy=target_energy_float,
            population_size=num_reads,
            rank=rank,
            compute_backend=compute_backend,
            job_metadata=job_metadata,
        )

        if sampler.type == "qasm":
            file_path = sampler.model.qasm_filepath + sampler.model.qasm_filename
            file_zip = sampler.filepath + sampler.filename + ".zip"
            with zipfile.ZipFile(file_zip, "w", zipfile.ZIP_DEFLATED) as f:
                f.write(file_path, arcname=sampler.model.qasm_filename)
        else:
            file_path = sampler.filepath + sampler.filename
            file_zip = sampler.filepath + sampler.filename + ".zip"
            with zipfile.ZipFile(file_zip, "w", zipfile.ZIP_DEFLATED) as f:
                f.write(file_path, arcname=sampler.filename)
        self.logger.info("Submitting the job to Dynex.")

        return self._get_grpc_client().create_job(
            opts=opts,
            file_zip=file_zip,
            job_filename=sampler.filename,
            retry_count=self.config.retry_count,
        )

    def create_job_api_proto(
        self,
        sampler: "DynexSampler",  # noqa: F821
        annealing_time: int,
        switchfraction: int,
        num_reads: int,
        alpha: int = 20,
        beta: int = 20,
        gamma: int = 1,
        delta: int = 1,
        epsilon: int = 1,
        zeta: int = 1,
        minimum_stepsize: float = 0.05,
        block_fee: int = 0,
        shots: int = 1,
        rank: int = 1,
        target_energy: float = 0.0,
        job_metadata: Optional[dict] = None,
        debugging: bool = False,
    ) -> tuple:
        """Create a new job via structured protobuf message (no file I/O on the hot path).

        Falls back to the file-based path for:
        - QASM-type samplers (require backend circuit conversion)
        - Cluster mode (multiple sub-jobs, handled separately)
        """
        if not self.config.mainnet:
            raise NotImplementedError("Job creation is only available in network mode")

        # Cluster and QASM paths still use file-based transport
        if sampler.type == "qasm" or isinstance(sampler.clauses, list):
            return self.create_job_api(
                sampler=sampler,
                annealing_time=annealing_time,
                switchfraction=switchfraction,
                num_reads=num_reads,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                delta=delta,
                epsilon=epsilon,
                zeta=zeta,
                minimum_stepsize=minimum_stepsize,
                block_fee=block_fee,
                shots=shots,
                rank=rank,
                target_energy=target_energy,
                job_metadata=job_metadata,
            )

        qubo, offset = sampler.clauses
        var_mappings = sampler.var_mappings  # {new_int: original_label}
        num_vars = sampler.num_variables

        # Build index arrays matching _save_wcnf's output exactly.
        # var_mappings maps new_int → original_label; invert it once for O(1) per-entry lookup.
        inv_mappings: dict = {mv: k for k, mv in var_mappings.items()} if var_mappings else {}

        rows: list = []
        cols: list = []
        vals: list = []
        for (i, j), v in qubo.items():
            rows.append(int(inv_mappings.get(i, i)))
            cols.append(int(inv_mappings.get(j, j)))
            vals.append(float(v))

        if debugging:
            import os
            os.makedirs(sampler.filepath, exist_ok=True)
            sampler._save_wcnf(
                sampler.clauses,
                sampler.filepath + sampler.filename,
                num_vars,
                sampler.num_clauses,
                var_mappings,
            )
            if self.logger:
                self.logger.debug(f"[debug] job file: {sampler.filepath + sampler.filename}")

        target_energy_float = float(target_energy) if hasattr(target_energy, "item") else float(target_energy)
        compute_backend = getattr(sampler.config, "compute_backend", "unspecified")

        opts = JobOptions(
            annealing_time=annealing_time,
            switchfraction=switchfraction,
            num_reads=num_reads,
            params=[alpha, beta, gamma, delta, epsilon, zeta],
            min_step_size=minimum_stepsize,
            description=sampler.description,
            block_fee=block_fee,
            shots=shots,
            target_energy=target_energy_float,
            population_size=num_reads,
            rank=rank,
            compute_backend=compute_backend,
            job_metadata=job_metadata,
        )

        if self.logger:
            self.logger.info("Submitting the job to Dynex.")
        return self._get_grpc_client().create_job_from_data(
            opts=opts,
            rows=rows,
            cols=cols,
            vals=vals,
            offset=float(offset),
            num_vars=num_vars,
            job_filename=sampler.filename,
            retry_count=self.config.retry_count,
        )


class JobOptions(BaseModel):
    """Job options for creating a new job on Dynex platform."""

    annealing_time: int = Field(..., ge=1, description="Maximum number of annealing steps")
    switchfraction: float = Field(default=0.0, ge=0.0, description="Switch fraction parameter")
    num_reads: int = Field(..., ge=1, description="Number of reads/samples")
    params: List[float] = Field(
        default_factory=lambda: [20, 20, 1, 1, 1, 1],
        description="Parameters [alpha, beta, gamma, delta, epsilon, zeta]",
    )
    min_step_size: float = Field(default=0.05, ge=0.0, le=1.0, description="Minimum step size")
    description: str = Field(default="", max_length=1000, description="Job description")
    block_fee: int = Field(default=0, ge=0, description="Block fee")
    shots: int = Field(default=1, ge=1, description="Number of shots")
    target_energy: float = Field(default=0.0, description="Target energy")
    population_size: int = Field(default=0, ge=0, description="Population size")
    rank: int = Field(default=1, ge=1, description="Rank parameter")
    reward: int = Field(default=0, ge=0, description="Reward amount")
    compute_backend: Union[ComputeBackend, str, int] = Field(
        default="unspecified", description="Compute backend type: 'unspecified', 'cpu', 'gpu', or 'qpu'"
    )
    request_ip: str = Field(default="", description="Request IP address")
    job_metadata: Optional[dict] = Field(
        default=None, description="Job metadata: {'type': 'qasm'} for Circuit BQM, None for Constraint BQM"
    )

    @field_validator("target_energy", mode="before")
    @classmethod
    def convert_target_energy(cls, v):
        """Convert numpy types to Python native types."""
        if hasattr(v, "item"):  # numpy scalar
            return float(v.item())
        return float(v)

    @field_validator("compute_backend", mode="before")
    @classmethod
    def validate_compute_backend(cls, v):
        """Validate and convert compute_backend to string."""
        if isinstance(v, int):
            # Map int values to string names
            mapping = {0: "unspecified", 1: "cpu", 2: "gpu", 3: "qpu"}
            return mapping.get(v, "unspecified")
        if isinstance(v, str):
            v_lower = v.lower()
            valid_values = ["unspecified", "cpu", "gpu", "qpu"]
            if v_lower not in valid_values:
                raise DynexValidationError(f"compute_backend must be one of {valid_values}, got '{v}'")
            return v_lower
        return "unspecified"

    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "JobOptions":
        """Create from dictionary."""
        return cls(**data)
