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

gRPC streaming / solution-delivery mixin for _DynexSampler.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import queue
import threading
import time
import urllib.request
from typing import Optional

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
from dynex.proto import sdk_pb2


class SolutionStreamingMixin:
    """Mixin providing gRPC streaming and solution-cache management for _DynexSampler."""

    # ------------------------------------------------------------------ #
    # State initialisation (called from _DynexSampler.__init__)           #
    # ------------------------------------------------------------------ #

    def _init_streaming_state(self) -> None:
        """Initialise all gRPC/solution-cache attributes."""
        self._downloaded_solutions: set = set()
        self._grpc_solution_queue: queue.Queue = queue.Queue()
        self._grpc_stream_thread: Optional[threading.Thread] = None
        self._grpc_stream_stop: threading.Event = threading.Event()
        self._grpc_stream_lock: threading.Lock = threading.Lock()
        self._grpc_active_call = None
        self._grpc_last_seq: int = 0
        self._grpc_subscription_disabled: bool = False
        self._grpc_subscription_bootstrap_done: bool = False
        self._grpc_solution_meta: dict = {}
        self._job_error: Optional[str] = None
        self._grpc_solution_stats: dict = {}
        self._grpc_solution_remote: dict = {}
        self._grpc_downloaded_files: set = set()
        self._solution_cache: dict = {}

    # ------------------------------------------------------------------ #
    # gRPC subscription lifecycle                                          #
    # ------------------------------------------------------------------ #

    def _stop_grpc_subscription(self) -> None:
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

    def _reset_grpc_subscription(self) -> None:
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

    def _ensure_grpc_subscription(self) -> None:
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

    # ------------------------------------------------------------------ #
    # gRPC streaming worker                                                #
    # ------------------------------------------------------------------ #

    def _grpc_solution_worker(self) -> None:
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
                        if self._protobuf_has_field(event, "error"):
                            error_info = event.error
                            error_code = getattr(error_info, "code", "UNKNOWN")
                            error_message = getattr(error_info, "message", "Job failed")
                            self._log_debug(
                                f"Received error event job_id={job_id} seq={event.seq} "
                                f"code={error_code} message={error_message}"
                            )
                            if self.logging:
                                self.logger.error(f"Job {job_id} failed: {error_message}")
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
                            kind = getattr(solution_payload, "kind", "")
                            has_data = hasattr(solution_payload, "data") and bool(getattr(solution_payload, "data", ""))
                            has_url = bool(getattr(solution_payload, "url", ""))
                            data_len = len(getattr(solution_payload, "data", "")) if has_data else 0
                            self._log_debug(
                                f"Queueing SOLUTION_NEW job_id={job_id} seq={event.seq} name={solution_name} "
                                f"kind={kind} has_data={has_data} has_url={has_url} data_len={data_len}"
                            )
                            if self.logging:
                                self.logger.debug(
                                    f"Received solution event job_id={job_id} name={solution_name} "
                                    f"kind={kind} has_inline={has_data} has_url={has_url} data_len={data_len}"
                                )
                            self._grpc_solution_queue.put(solution_payload)
                        else:
                            self._log_debug(
                                f"SOLUTION_NEW event has no envelope or solution payload "
                                f"job_id={job_id} seq={event.seq}"
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

    # ------------------------------------------------------------------ #
    # Solution metadata consumption                                        #
    # ------------------------------------------------------------------ #

    def _consume_solution_meta(self, solution) -> None:
        if self.logger and self.logger.isEnabledFor(logging.DEBUG):
            try:
                fields = {desc.name: value for desc, value in solution.ListFields()}  # type: ignore[attr-defined]
            except AttributeError:
                fields = {}
            else:
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

        if remote_name and ("," in remote_name or len(remote_name) > 200):
            if checksum:
                safe_name = self._sanitize_solution_name(checksum)
                if self.logging:
                    self.logger.warning(
                        f"remote_name appears to be solution data (length={len(remote_name)}), "
                        f"using checksum instead: {checksum[:20]}..."
                    )
            else:
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
        inline_data = ""
        if hasattr(solution, "data"):
            inline_data = getattr(solution, "data", "") or ""
        subject = getattr(solution, "subject", "")
        self._log_debug(
            f"Processing solution meta job_id={self.current_job_id} name={remote_name} "
            f"checksum={checksum} size={size_hint} valid={valid_flag} kind={kind} "
            f"url={bool(url)} inline={bool(inline_data)} safe={safe_name} subject={subject!r}"
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

        max_name_length = 200
        if len(safe_name) > max_name_length:
            name_hash = hashlib.md5(safe_name.encode("utf-8")).hexdigest()[:8]
            safe_name = safe_name[: max_name_length - 9] + "_" + name_hash
            if self.logging:
                self.logger.warning(f"safe_name too long ({len(safe_name)}), truncated to: {safe_name[:50]}...")

        local_name = f"{self.filename}.{safe_name}"
        local_path = os.path.join(self.filepath, local_name)

        if len(local_path) > 250:
            path_hash = hashlib.md5(local_path.encode("utf-8")).hexdigest()[:16]
            local_name = f"{self.filename}.sol_{path_hash}"
            local_path = os.path.join(self.filepath, local_name)
            if self.logging:
                self.logger.warning(f"Local path too long, using hash-based name: {local_name}")

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

        raw_data: Optional[bytes] = None

        if inline_data and kind == "inline":
            try:
                self._log_debug(
                    f"Processing inline solution data name={remote_name} kind={kind} " f"data_len={len(inline_data)}"
                )
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

        if raw_data is None and url:
            try:
                self._log_debug(f"Downloading solution via presigned URL name={remote_name} url={url[:50]}...")
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
            f"Solution stored name={remote_name} key={local_name} "
            f"downloaded_count={len(self._downloaded_solutions)}"
        )

    # ------------------------------------------------------------------ #
    # File listing                                                         #
    # ------------------------------------------------------------------ #

    def list_files_with_text(self) -> None:
        """Fetch assignment files from gRPC and update internal solution tracking."""
        if self.config.mainnet:
            self._list_files_with_text_grpc()

    def list_files_with_text_local(self) -> list:
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

    def _list_files_with_text_ftp(self) -> None:
        """FTP transport removed - using gRPC only."""
        pass

    def _list_files_with_text_grpc(self) -> None:
        if self.current_job_id is None:
            return

        if self._grpc_subscription_disabled:
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

        if not drained:
            self._grpc_subscription_bootstrap_done = True

    # ------------------------------------------------------------------ #
    # File cleanup                                                         #
    # ------------------------------------------------------------------ #

    def delete_local_files_by_prefix(self, directory: str, prefix: str) -> None:
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

    def _delete_files_by_prefix(self, directory: str, prefix: str) -> None:
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

    # ------------------------------------------------------------------ #
    # Solution stats helpers                                               #
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    # Static method aliases (accessible as class attributes)              #
    # ------------------------------------------------------------------ #

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
