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

Solution metadata parsing and decoding helpers.

All functions here are stateless / pure, so they can be tested and reused
independently of the sampler lifecycle.
"""

from __future__ import annotations

import io
import json
import re
from typing import NamedTuple, Optional

from dynex.exceptions import DynexJobError

try:
    import zstandard as zstd
except ModuleNotFoundError:
    zstd = None


class SolutionMetrics(NamedTuple):
    chips: int
    steps: int
    loc: int
    energy: float


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------

def coerce_int(value) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def coerce_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Name / subject / stats parsing
# ---------------------------------------------------------------------------

def sanitize_solution_name(name: str) -> str:
    if not name:
        return "solution"
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return sanitized or "solution"


def parse_solution_subject(subject: str) -> dict:
    if not subject:
        return {}
    try:
        data = json.loads(subject)
        if isinstance(data, dict):
            return data
    except (TypeError, ValueError):
        pass
    tokens = re.split(r"[;,]\s*", subject)
    data: dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            data[key] = value
    return data


def parse_solution_numbers(text: str) -> dict:
    if not text:
        return {}
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(matches) < 4:
        return {}
    result: dict[str, float] = {}
    try:
        result["chips"] = int(float(matches[0]))
        result["steps"] = int(float(matches[1]))
        result["loc"] = int(float(matches[2]))
        result["energy"] = float(matches[3])
    except ValueError:
        return {}
    return result


def extract_solution_stats(solution, remote_name: str) -> dict:
    """Build stats dict from a protobuf-like solution object."""
    subject = getattr(solution, "subject", "")
    subject_stats_raw = parse_solution_subject(subject)
    subject_stats = dict(subject_stats_raw) if subject_stats_raw else {}
    stats: dict[str, object] = {}
    if subject_stats:
        stats.update(subject_stats)
        data_field = (
            subject_stats.get("data")
            or subject_stats.get("payload")
            or subject_stats.get("inline_data")
        )
        if data_field:
            stats["inline_data"] = data_field
    numeric_stats = {}
    for key in ("chips", "steps", "loc", "energy"):
        if key in stats:
            continue
        value = subject_stats.get(key)
        if value is None:
            continue
        numeric_stats[key] = value
    if not numeric_stats:
        numeric_stats = parse_solution_numbers(remote_name)
    stats.update(numeric_stats)
    if subject_stats:
        subject_stats = dict(subject_stats)
        subject_stats.pop("data", None)
        subject_stats.pop("payload", None)
    stats["subject_dict"] = subject_stats
    return stats


def solution_metrics_from_filename(
    filename: str,
    fallback_info: str,
    stats: dict,
) -> SolutionMetrics:
    chips = coerce_int(stats.get("chips")) if stats else None
    steps = coerce_int(stats.get("steps")) if stats else None
    loc = coerce_int(stats.get("loc")) if stats else None
    energy = coerce_float(stats.get("energy")) if stats else None

    if chips is None or steps is None or loc is None or energy is None:
        parsed = parse_solution_numbers(fallback_info)
        if chips is None:
            chips = parsed.get("chips")
        if steps is None:
            steps = parsed.get("steps")
        if loc is None:
            loc = parsed.get("loc")
        if energy is None:
            energy = parsed.get("energy")

    return SolutionMetrics(
        chips=coerce_int(chips) or 0,
        steps=coerce_int(steps) or 0,
        loc=coerce_int(loc) or 0,
        energy=coerce_float(energy) or 0.0,
    )


# ---------------------------------------------------------------------------
# Binary / protobuf helpers
# ---------------------------------------------------------------------------

def decompress_bytes(data: bytes, compression: Optional[str]) -> bytes:
    if not compression:
        return data
    compression = compression.lower()
    if compression == "zstd":
        if zstd is None:
            raise ModuleNotFoundError(
                "zstandard is required to decode zstd-compressed solutions"
            )
        decompressor = zstd.ZstdDecompressor()
        try:
            return decompressor.decompress(data)
        except zstd.ZstdError:
            with decompressor.stream_reader(io.BytesIO(data)) as reader:
                return reader.read()
    return data


def decode_varint(buffer: bytes, index: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        if index >= len(buffer):
            raise DynexJobError("truncated varint while decoding solution payload")
        byte = buffer[index]
        index += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return result, index
        shift += 7
        if shift >= 64:
            raise DynexJobError("varint overflow while decoding solution payload")


def skip_field(buffer: bytes, index: int, wire_type: int) -> int:
    if wire_type == 0:
        _, index = decode_varint(buffer, index)
    elif wire_type == 1:
        index += 8
    elif wire_type == 2:
        length, index = decode_varint(buffer, index)
        index += length
    elif wire_type == 3:
        index = skip_group(buffer, index)
    elif wire_type == 4:
        pass
    elif wire_type == 5:
        index += 4
    else:
        raise DynexJobError(
            f"unknown wire type {wire_type} while decoding solution payload"
        )
    return index


def skip_group(buffer: bytes, index: int) -> int:
    while True:
        tag, index = decode_varint(buffer, index)
        wire_type = tag & 0x7
        if wire_type == 4:
            return index
        index = skip_field(buffer, index, wire_type)


def protobuf_has_field(message, field_name: str) -> bool:
    descriptor = getattr(message, "DESCRIPTOR", None)
    if descriptor is None or field_name not in descriptor.fields_by_name:
        return False
    try:
        return message.HasField(field_name)
    except ValueError:
        return False
