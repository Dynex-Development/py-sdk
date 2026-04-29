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

Local solver I/O mixin: file-save helpers and local solution reading.
"""

from __future__ import annotations

import os
import secrets
from typing import Optional

from dynex._voltage import ensure_voltage_text, extract_voltage_values, process_voltage_line


class LocalSolverMixin:
    """Mixin providing file I/O helpers and local solver support for _DynexSampler."""

    # ------------------------------------------------------------------ #
    # Static helpers                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _check_list_length(lst: list) -> bool:
        """Returns True if the sat problem is k-Sat (any clause has > 3 literals)."""
        for sublist in lst:
            if isinstance(sublist, list) and len(sublist) > 3:
                return True
        return False

    @staticmethod
    def _save_cnf(clauses: list, filename: str) -> None:
        """Save problem as CNF file."""
        num_variables = max(max(abs(lit) for lit in clause) for clause in clauses)
        num_clauses = len(clauses)
        with open(filename, "w") as f:
            f.write("p cnf %d %d\n" % (num_variables, num_clauses))
            for clause in clauses:
                f.write(" ".join(str(int(lit)) for lit in clause) + " 0\n")

    def _save_wcnf(
        self,
        clauses,
        filename: str,
        num_variables: int,
        num_clauses: int,
        var_mappings: Optional[dict],
    ) -> None:
        """Save problem as WCNF file."""
        # Build inverse mapping once (O(n)) rather than scanning per entry (O(n²)).
        inv_mappings: dict = {v: k for k, v in var_mappings.items()} if var_mappings else {}
        with open(filename, "w") as f:
            f.write("p qubo %d %d %f\n" % (num_variables, num_clauses, clauses[1]))
            for (i, j), value in clauses[0].items():
                i = inv_mappings.get(i, i)
                j = inv_mappings.get(j, j)
                f.write("%d %d %f\n" % (i, j, value))

    # ------------------------------------------------------------------ #
    # Local filesystem helpers                                             #
    # ------------------------------------------------------------------ #

    def add_salt_local(self) -> None:
        """Rename solution files to add a random hex suffix for uniqueness."""
        directory = self.filepath_full
        fn = self.filename + "."
        for filename in os.listdir(directory):
            if filename.startswith(fn):
                if filename.split(".")[-1].isnumeric():
                    os.rename(
                        directory + "/" + filename,
                        directory + "/" + filename + "." + secrets.token_hex(16),
                    )

    # ------------------------------------------------------------------ #
    # Voltage / solution reading                                           #
    # ------------------------------------------------------------------ #

    def read_voltage_data(self, file, mainnet: bool, rank: int):
        """Read solution data from in-memory cache (mainnet) or disk (local)."""
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
        skip_first = self.type == "qasm"
        return self._extract_voltage_values(data, prefer_last=True, skip_first=skip_first)

    def _read_second_line(self, file_obj):
        data = file_obj.read()
        skip_first = self.type == "qasm"
        return self._extract_voltage_values(data, prefer_last=False, skip_first=skip_first)

    def _read_entire_file(self, file_obj):
        data = file_obj.read()
        skip_first = self.type == "qasm"
        return self._extract_voltage_values(data, prefer_last=False, skip_first=skip_first)

    # Static aliases for voltage helpers so they are accessible as class attributes.
    _extract_voltage_values = staticmethod(extract_voltage_values)
    _process_voltage_line = staticmethod(process_voltage_line)
    _ensure_voltage_text = staticmethod(ensure_voltage_text)
