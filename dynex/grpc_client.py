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

from __future__ import annotations

import io
import json
import logging
import os
import sys
import tempfile
from typing import TYPE_CHECKING, Iterable, NamedTuple, Optional, Tuple, Union
from urllib.parse import urlparse

import grpc

from dynex.exceptions import DynexConnectionError, DynexJobError, DynexValidationError
from dynex.proto import sdk_pb2, sdk_pb2_grpc

if TYPE_CHECKING:
    from dynex.api import JobOptions
    from dynex.config import DynexConfig


class JobCreationResult(NamedTuple):
    job_id: int
    filename: str
    price_per_block: float
    qasm: dict | None


def _qubo_arrays_to_wcnf_bytes(
    rows: list,
    cols: list,
    vals: list,
    offset: float,
    num_vars: int,
) -> bytes:
    """Build WCNF text matching QRE ReconstructWCNF / legacy chunk pipeline."""
    n = len(rows)
    buf = io.StringIO()
    buf.write(f"p qubo {int(num_vars)} {n} {float(offset):g}\n")
    for r, c, v in zip(rows, cols, vals):
        buf.write(f"{int(r)} {int(c)} {float(v):g}\n")
    return buf.getvalue().encode("utf-8")


class DynexGrpcClient:
    """Encapsulates gRPC interactions with the Dynex backend."""

    _CHUNK_SIZE = 4 * 1024 * 1024

    # Mapping from string compute backend names to protobuf enum values
    _COMPUTE_BACKEND_MAPPING: dict[str, int] = {
        "unspecified": sdk_pb2.COMPUTE_BACKEND_UNSPECIFIED,
        "cpu": sdk_pb2.COMPUTE_BACKEND_DYNEX_CPU,
        "gpu": sdk_pb2.COMPUTE_BACKEND_DYNEX_GPU,
        "qpu": sdk_pb2.COMPUTE_BACKEND_DYNEX_QPU_APOLLO,
    }

    def __init__(self, config: "DynexConfig", logger: Optional[logging.Logger] = None) -> None:
        self.config = config
        self.logger = logger
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[sdk_pb2_grpc.SDKStub] = None

    def _log_info(self, message: str) -> None:
        if self.logger:
            self.logger.info(message)

    def _log_error(self, message: str) -> None:
        if self.logger:
            self.logger.error(message)

    def _log_debug(self, message: str) -> None:
        if self.logger and self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(message)

    def _log_success(self, message: str) -> None:
        """Log success message with green color"""
        if self.logger:
            if sys.stdout.isatty():
                colored_message = f"\033[92mSUCCESS: {message}\033[0m"
            else:
                colored_message = f"SUCCESS: {message}"
            self.logger.info(colored_message)

    def _log_warning(self, message: str) -> None:
        """Log warning message with yellow color"""
        if self.logger:
            if sys.stdout.isatty():
                colored_message = f"\033[93m{message}\033[0m"
            else:
                colored_message = message
            self.logger.warning(colored_message)

    def _log_progress(self, message: str) -> None:
        """Log progress message with blue color"""
        if self.logger:
            if sys.stdout.isatty():
                colored_message = f"\033[94mPROGRESS: {message}\033[0m"
            else:
                colored_message = f"PROGRESS: {message}"
            self.logger.info(colored_message)

    def _log_grpc_action(self, action: str, details: str = "") -> None:
        """Log gRPC action with consistent formatting"""
        if self.logger:
            if sys.stdout.isatty():
                formatted_message = f"\033[96mgRPC: {action}\033[0m"
                if details:
                    formatted_message += f" \033[90m{details}\033[0m"
            else:
                formatted_message = f"gRPC: {action}"
                if details:
                    formatted_message += f" {details}"
            self.logger.info(formatted_message)

    def _metadata(self) -> Tuple[Tuple[str, str], ...]:
        return (("authorization", f"Bearer {self.config.sdk_key}"),)

    def _get_stub(self) -> sdk_pb2_grpc.SDKStub:
        if self._stub is not None:
            return self._stub

        # Use dedicated GRPC_ENDPOINT if provided
        grpc_endpoint = self.config.grpc_endpoint
        if grpc_endpoint:
            # GRPC_ENDPOINT is typically just "host:port" without protocol
            if "://" in grpc_endpoint:
                parsed = urlparse(grpc_endpoint)
                host = parsed.hostname or parsed.path
                port = parsed.port or 3000
                use_tls = parsed.scheme == "https"
            else:
                # Format is "host:port"
                if ":" in grpc_endpoint:
                    host, port_str = grpc_endpoint.rsplit(":", 1)
                    port = int(port_str)
                else:
                    host = grpc_endpoint
                    port = 3000
                # Determine TLS based on host (localhost = no TLS, otherwise TLS)
                use_tls = host not in ("localhost", "127.0.0.1")
        else:
            # Fallback: derive from gRPC endpoint
            endpoint = self.config.grpc_endpoint
            if not endpoint:
                raise DynexValidationError("GRPC_ENDPOINT is not configured")
            if "://" not in endpoint:
                endpoint = f"https://{endpoint}"
            parsed = urlparse(endpoint)
            host = parsed.hostname or parsed.path
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            use_tls = parsed.scheme == "https"

        target = f"{host}:{port}"

        # Increase max message size to 50MB for large BQMs (e.g., 110-Queens, 150-Queens)
        max_msg_size = 50 * 1024 * 1024  # 50 MB
        options = [
            ("grpc.max_receive_message_length", max_msg_size),
            ("grpc.max_send_message_length", max_msg_size),
        ]

        if use_tls:
            channel = grpc.secure_channel(target, grpc.ssl_channel_credentials(), options=options)
        else:
            channel = grpc.insecure_channel(target, options=options)

        self._channel = channel
        self._stub = sdk_pb2_grpc.SDKStub(channel)
        return self._stub

    def _resolve_rpc(
        self,
        stub: sdk_pb2_grpc.SDKStub,
        method_name: str,
        rpc_kind: str,
        path: str,
        request_serializer,
        response_deserializer,
    ):
        rpc = getattr(stub, method_name, None)
        if rpc is not None:
            return rpc
        channel = self._channel
        if channel is None:
            raise NotImplementedError(f"{method_name} RPC is not available on this server")
        factory = getattr(channel, rpc_kind, None)
        if factory is None:
            raise NotImplementedError(f"Channel does not support RPC kind {rpc_kind} for {method_name}")
        return factory(
            path,
            request_serializer=request_serializer,
            response_deserializer=response_deserializer,
        )

    def _build_job_options(self, opts: Union["JobOptions", dict]) -> sdk_pb2.JobNewOpts:
        """Translate JobOptions or legacy opts dict into structured JobNewOpts message."""
        if hasattr(opts, "model_dump"):
            opts_dict = opts.model_dump()
        else:
            opts_dict = opts

        compute_backend_raw = opts_dict.get("compute_backend", "unspecified")
        if isinstance(compute_backend_raw, str):
            compute_backend_value = self._COMPUTE_BACKEND_MAPPING.get(
                compute_backend_raw.lower(), sdk_pb2.COMPUTE_BACKEND_UNSPECIFIED
            )
        elif isinstance(compute_backend_raw, int):
            compute_backend_value = compute_backend_raw
        else:
            compute_backend_value = sdk_pb2.COMPUTE_BACKEND_UNSPECIFIED

        if "service_type" in opts_dict:
            self._log_debug("Ignoring deprecated service_type option")
        if "use_gpu" in opts_dict:
            self._log_debug("Ignoring deprecated use_gpu option")

        job_metadata = opts_dict.get("job_metadata")
        metadata_json = json.dumps(job_metadata) if job_metadata else ""

        return sdk_pb2.JobNewOpts(
            max_steps=int(opts_dict.get("annealing_time", 0)),
            num_reads=int(opts_dict.get("num_reads", 0)),
            min_step_size=float(opts_dict.get("min_step_size", 0.0)),
            description=str(opts_dict.get("description", "")),
            block_fee=int(opts_dict.get("block_fee", 0)),
            reward=int(opts_dict.get("reward", 0)),
            shots=int(opts_dict.get("shots", 0)),
            target_energy=float(opts_dict.get("target_energy", 0.0)),
            population_size=int(opts_dict.get("population_size", 0)),
            rank=int(opts_dict.get("rank", 0)),
            compute_backend=compute_backend_value,
            job_metadata_json=metadata_json,
        )

    def _iter_create_job_requests(
        self, opts: Union["JobOptions", dict], file_zip: str
    ) -> Iterable[sdk_pb2.CreateJobRequest]:
        # Handle Pydantic model
        if hasattr(opts, "model_dump"):
            opts_dict = opts.model_dump()
            request_ip = opts.request_ip
        else:
            opts_dict = opts
            request_ip = opts_dict.get("request_ip", "")

        init_payload = sdk_pb2.CreateJobInit(
            opts=self._build_job_options(opts),
            request_ip=str(request_ip),
        )
        yield sdk_pb2.CreateJobRequest(init=init_payload)
        with open(file_zip, "rb") as fh:
            while True:
                chunk = fh.read(DynexGrpcClient._CHUNK_SIZE)
                if not chunk:
                    break
                yield sdk_pb2.CreateJobRequest(chunk=sdk_pb2.JobChunk(data=chunk))

    def create_job(
        self, opts: Union["JobOptions", dict], file_zip: str, job_filename: str, retry_count: int
    ) -> JobCreationResult:
        last_exception: Optional[Exception] = None

        for try_count in range(retry_count, 0, -1):
            try:
                stub = self._get_stub()
                response = stub.CreateJob(
                    self._iter_create_job_requests(opts, file_zip),
                    metadata=self._metadata(),
                )
                qasm = json.loads(response.qasm_json) if response.qasm_json else None
                self._log_success(f"Job created successfully (job_id={response.job_id})")
                return JobCreationResult(
                    job_id=response.job_id,
                    filename=job_filename,
                    price_per_block=response.real_price_per_block,
                    qasm=qasm,
                )
            except grpc.RpcError as e:
                last_exception = e
                self._log_error(f"gRPC request failed: {e}")
                if try_count > 1:
                    self._log_warning(f"Retrying... ({try_count - 1} attempts left)")
                else:
                    raise DynexConnectionError(f"gRPC job creation failed: {e}") from e
            except Exception as e:
                last_exception = e
                self._log_error(f"Unexpected error: {e}")
                if try_count > 1:
                    self._log_warning(f"Retrying... ({try_count - 1} attempts left)")
                else:
                    raise DynexJobError(f"Job creation failed: {e}") from e

        raise DynexJobError(
            f"Job creation failed after {retry_count} attempts: {str(last_exception)}"
        ) from last_exception

    def _iter_create_job_from_data_requests(
        self,
        opts: Union["JobOptions", dict],
        rows: list,
        cols: list,
        vals: list,
        offset: float,
        num_vars: int,
        filename: str,
    ) -> Iterable[sdk_pb2.CreateJobRequest]:
        if hasattr(opts, "model_dump"):
            request_ip = opts.request_ip
        else:
            request_ip = opts.get("request_ip", "") if isinstance(opts, dict) else ""

        init_payload = sdk_pb2.CreateJobInit(
            opts=self._build_job_options(opts),
            request_ip=str(request_ip),
        )
        yield sdk_pb2.CreateJobRequest(init=init_payload)

        job_data_msg = sdk_pb2.JobData(
            num_vars=int(num_vars),
            row=rows,
            col=cols,
            val=vals,
            offset=float(offset),
            filename=filename,
        )
        yield sdk_pb2.CreateJobRequest(job_data=job_data_msg)

    def _create_job_via_wcnf_chunks(
        self,
        opts: Union["JobOptions", dict],
        rows: list,
        cols: list,
        vals: list,
        offset: float,
        num_vars: int,
        job_filename: str,
        retry_count: int,
    ) -> JobCreationResult:
        """Legacy QRE without job_data oneof: upload the same QUBO as WCNF file chunks."""
        payload = _qubo_arrays_to_wcnf_bytes(rows, cols, vals, offset, num_vars)
        fd, path = tempfile.mkstemp(suffix=".wcnf", prefix="dynex_job_")
        closed = False
        try:
            os.write(fd, payload)
            os.close(fd)
            closed = True
            return self.create_job(opts, path, job_filename, retry_count)
        finally:
            if not closed:
                try:
                    os.close(fd)
                except OSError:
                    pass
            try:
                os.unlink(path)
            except OSError:
                pass

    def create_job_from_data(
        self,
        opts: Union["JobOptions", dict],
        rows: list,
        cols: list,
        vals: list,
        offset: float,
        num_vars: int,
        job_filename: str,
        retry_count: int,
    ) -> JobCreationResult:
        last_exception: Optional[Exception] = None

        for try_count in range(retry_count, 0, -1):
            try:
                stub = self._get_stub()
                response = stub.CreateJob(
                    self._iter_create_job_from_data_requests(opts, rows, cols, vals, offset, num_vars, job_filename),
                    metadata=self._metadata(),
                )
                qasm = json.loads(response.qasm_json) if response.qasm_json else None
                self._log_success(f"Job created successfully (job_id={response.job_id})")
                return JobCreationResult(
                    job_id=response.job_id,
                    filename=job_filename,
                    price_per_block=response.real_price_per_block,
                    qasm=qasm,
                )
            except grpc.RpcError as e:
                details = (e.details() or "").lower()
                if e.code() == grpc.StatusCode.INVALID_ARGUMENT and "unsupported payload" in details:
                    self._log_warning("Server does not accept job_data (legacy QRE); falling back to WCNF chunk upload")
                    return self._creatmake_job_via_wcnf_chunks(
                        opts, rows, cols, vals, offset, num_vars, job_filename, retry_count
                    )
                last_exception = e
                self._log_error(f"gRPC request failed: {e}")
                if try_count > 1:
                    self._log_warning(f"Retrying... ({try_count - 1} attempts left)")
                else:
                    raise DynexConnectionError(f"gRPC job creation failed: {e}") from e
            except Exception as e:
                last_exception = e
                self._log_error(f"Unexpected error: {e}")
                if try_count > 1:
                    self._log_warning(f"Retrying... ({try_count - 1} attempts left)")
                else:
                    raise DynexJobError(f"Job creation failed: {e}") from e

        raise DynexJobError(
            f"Job creation failed after {retry_count} attempts: {str(last_exception)}"
        ) from last_exception

    def update_job(self, job_id: int) -> sdk_pb2.UpdateJobReply:
        self._log_grpc_action("Updating job", f"job_id={job_id}")
        stub = self._get_stub()
        response = stub.UpdateJob(
            sdk_pb2.UpdateJobRequest(job_id=job_id),
            metadata=self._metadata(),
        )
        self._log_success(f"Job updated (job_id={job_id})")
        return response

    def cancel_job(self, job_id: int) -> sdk_pb2.CancelJobReply:
        stub = self._get_stub()
        response = stub.CancelJob(
            sdk_pb2.CancelJobRequest(job_id=job_id),
            metadata=self._metadata(),
        )
        self._log_success(f"Job cancelled (job_id={response.job_id})")
        return response

    def finish_job(self, job_id: int, min_loc: float, min_energy: float) -> sdk_pb2.FinishJobReply:
        stub = self._get_stub()
        # Limit min_loc to int32 range to avoid overflow
        min_loc_int = int(min_loc) if min_loc <= 2147483647 else 2147483647
        response = stub.FinishJob(
            sdk_pb2.FinishJobRequest(job_id=job_id, min_loc=min_loc_int, min_energy=min_energy),
            metadata=self._metadata(),
        )
        self._log_success(f"Job finished (job_id={response.job_id})")
        return response

    def download_solution(self, job_id: int, name: str, destination_path: str) -> None:
        self._log_grpc_action("Downloading solution", f"job_id={job_id}, name={name}")
        stub = self._get_stub()
        self._log_debug(f"gRPC DownloadSolution start job_id={job_id} name={name} destination={destination_path}")
        download_rpc = self._resolve_rpc(
            stub,
            "DownloadSolution",
            "unary_stream",
            "/dynex.sdk.v2.SDK/DownloadSolution",
            sdk_pb2.DownloadSolutionRequest.SerializeToString,
            sdk_pb2.SolutionChunk.FromString,
        )
        stream = download_rpc(
            sdk_pb2.DownloadSolutionRequest(job_id=job_id, name=name),
            metadata=self._metadata(),
        )
        directory = os.path.dirname(destination_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        chunk_count = 0
        total_bytes = 0
        with open(destination_path, "wb") as fh:
            for chunk in stream:
                data = chunk.data
                fh.write(data)
                chunk_count += 1
                total_bytes += len(data)
        self._log_success(f"Solution downloaded (job_id={job_id}, name={name}, size={total_bytes} bytes)")
        self._log_debug(
            f"gRPC DownloadSolution done job_id={job_id} name={name} chunks={chunk_count} bytes={total_bytes}"
        )

    def get_solution_url(self, job_id: int, name: str) -> Tuple[str, int]:
        stub = self._get_stub()
        get_url_rpc = self._resolve_rpc(
            stub,
            "GetSolutionURL",
            "unary_unary",
            "/dynex.sdk.v2.SDK/GetSolutionURL",
            sdk_pb2.GetSolutionURLRequest.SerializeToString,
            sdk_pb2.GetSolutionURLReply.FromString,
        )
        reply = get_url_rpc(
            sdk_pb2.GetSolutionURLRequest(job_id=job_id, name=name),
            metadata=self._metadata(),
        )
        return reply.url, reply.ttl_seconds

    def list_atomics(self, job_id: int, limit: Optional[int] = None) -> Iterable[sdk_pb2.AtomicForJob]:
        self._log_grpc_action("Listing atomics", f"job_id={job_id}, limit={limit}")
        stub = self._get_stub()
        request_kwargs = {"job_id": job_id}
        if limit is not None and limit > 0:
            request_kwargs["limit"] = int(limit)
        list_atomics_rpc = self._resolve_rpc(
            stub,
            "ListAtomics",
            "unary_unary",
            "/dynex.sdk.v2.SDK/ListAtomics",
            sdk_pb2.ListAtomicsRequest.SerializeToString,
            sdk_pb2.ListAtomicsReply.FromString,
        )
        response = list_atomics_rpc(
            sdk_pb2.ListAtomicsRequest(**request_kwargs),
            metadata=self._metadata(),
        )
        atomics_count = len(response.items)
        self._log_success(f"Atomics listed (job_id={job_id}, count={atomics_count})")
        return response.items

    def subscribe_job(self, job_id: int, from_seq: int = 0) -> grpc.Call:
        """Start a server-streaming SubscribeJob RPC."""
        stub = self._get_stub()
        self._log_debug(f"gRPC SubscribeJob start job_id={job_id} from_seq={from_seq}")
        return stub.SubscribeJob(
            sdk_pb2.SubscribeJobRequest(job_id=job_id, from_seq=from_seq),
            metadata=self._metadata(),
        )
