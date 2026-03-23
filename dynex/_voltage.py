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

Voltage data reading and parsing helpers.

Stateless functions for extracting solver voltages from raw solution
bytes or text lines.
"""

from __future__ import annotations

import re
from typing import List

from dynex._solution_parser import decompress_bytes


def ensure_voltage_text(line) -> str:
    """Convert raw bytes/str to a UTF-8 string, decompressing zstd if needed."""
    if not line:
        return ""
    if isinstance(line, str):
        return line

    data = bytes(line)
    if data.startswith(b"\x28\xb5/\xfd"):
        try:
            data = decompress_bytes(data, "zstd")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("zstandard is required to decode zstd-compressed solution results") from exc

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="ignore")


def extract_voltage_values(
    line,
    prefer_last: bool = False,
    skip_first: bool = False,
) -> List[str]:
    """Parse voltage CSV values from raw solution data."""
    text_line = ensure_voltage_text(line)
    if not text_line:
        return ["NaN"]

    stripped = text_line.strip()
    if stripped.startswith('"') and stripped.endswith('"') and len(stripped) >= 2:
        stripped = stripped[1:-1]

    lines = [entry.strip() for entry in stripped.splitlines() if entry.strip()]
    data_lines = [entry for entry in lines if "," in entry]
    if not data_lines:
        return ["NaN"]

    if skip_first and len(data_lines) > 1:
        data_lines = data_lines[1:]

    target_line = data_lines[-1] if prefer_last else data_lines[0]
    voltages = [value.strip() for value in re.split(r",\s*", target_line) if value.strip()]
    return voltages if voltages else ["NaN"]


def process_voltage_line(line) -> List[str]:
    return extract_voltage_values(line, prefer_last=False)
