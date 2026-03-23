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

from abc import ABC
from dataclasses import dataclass

import dimod
import numpy as np

from dynex.config import DynexConfig
from dynex.exceptions import DynexModelError


@dataclass
class ConversionResult:
    clauses: list
    num_variables: int
    num_clauses: int
    var_mappings: dict
    precision: float
    bqm: dimod.BinaryQuadraticModel
    wcnf_offset: float = 0.0


class DynexModel(ABC):
    """
    Abstract base class for Dynex models.
    Includes config and logger initialization, and conversion utilities.
    """

    def __init__(self, config: DynexConfig = None, logging: bool = False):
        self.config = config if config is not None else DynexConfig()
        self.logger = getattr(self.config, "logger", None)
        self.logging = logging
        self.type = "Unknown"
        self.type_str = "Unknown"

    def __str__(self) -> str:
        return self.type_str

    @staticmethod
    def _max_precision(bqm: dimod.BinaryQuadraticModel) -> float:
        """
        Returns the maximum precision for BQM conversion.

        Args:
            bqm: dimod.BinaryQuadraticModel

        Returns:
            float: precision value
        """
        # avoids deprecated to_numpy_matrix()
        linear_max = np.max(np.abs(list(bqm.linear.values()))) if bqm.linear else 0.0
        quadratic_max = np.max(np.abs(list(bqm.quadratic.values()))) if bqm.quadratic else 0.0
        max_abs_coeff = max(linear_max, quadratic_max)

        if max_abs_coeff == 0:
            raise DynexModelError("At least one weight must be > 0.0")
        precision = 10 ** (np.floor(np.log10(max_abs_coeff)) - 4)
        return precision

    def _convert_bqm_to_qubo(self, bqm: dimod.BinaryQuadraticModel, logging: bool = True) -> ConversionResult:
        mappings = bqm.variables._relabel_as_integers()
        clauses = []
        Q = bqm.to_qubo()
        Q_list = list(Q[0])
        if logging and self.logger:
            self.logger.info("Model converted to QUBO")

        newQ = []
        for i in range(0, len(Q_list)):
            touple = Q_list[i]
            w = Q[0][touple]
            newQ.append(w)
        max_abs_coeff = np.max(np.abs(newQ))
        if max_abs_coeff == 0:
            if self.logger:
                self.logger.error("At least one weight must be > 0.0")
            raise DynexModelError("At least one weight must be > 0.0")

        precision = 10 ** (np.floor(np.log10(max_abs_coeff)) - 4)

        if precision > 1:
            if logging and self.logger:
                self.logger.warning(f"Precision cut from {precision} to 1")
            precision = 1

        W_add = Q[1]
        if logging and self.logger:
            self.logger.info(f"QUBO: Constant offset of the binary quadratic model: {W_add}")

        for i in range(0, len(Q_list)):
            touple = Q_list[i]
            i_val = int(touple[0]) + 1
            j_val = int(touple[1]) + 1
            w = Q[0][touple]
            w_int = int(np.round(w / precision))

            if i_val == j_val:
                if w_int > 0:
                    clauses.append([w_int, -i_val])
                if w_int < 0:
                    clauses.append([-w_int, i_val])
            else:
                if w_int > 0:
                    clauses.append([w_int, -i_val, -j_val])
                if w_int < 0:
                    clauses.append([-w_int, i_val, -j_val])
                    clauses.append([-w_int, j_val])

        num_variables = len(bqm.variables)
        num_clauses = len(clauses)
        bqm.variables._relabel(mappings)
        return ConversionResult(
            clauses=clauses,
            num_variables=num_variables,
            num_clauses=num_clauses,
            var_mappings=mappings,
            precision=precision,
            bqm=bqm,
        )

    def _convert_bqm_to_qubo_direct(self, bqm: dimod.BinaryQuadraticModel, logging: bool = True) -> ConversionResult:
        mappings = bqm.variables._relabel_as_integers()
        clauses = []
        linear = [v for i, v in sorted(bqm.linear.items(), key=lambda x: x[0])]
        quadratic = [[i, j, v] for (i, j), v in bqm.quadratic.items()]
        precision = self._max_precision(bqm)
        if precision > 1:
            if logging and self.logger:
                self.logger.warning(f"Precision cut from {precision} to 1")
            precision = 1
        if logging and self.logger:
            self.logger.info(f"Precision set to {precision}")
        wcnf_offset = 0
        for i, w in enumerate(linear):
            weight = np.round(w / precision)
            if weight > 0:
                clauses.append([weight, -(i + 1)])
            elif weight < 0:
                clauses.append([-weight, (i + 1)])
                wcnf_offset += weight
        num_variables = len(linear)
        if quadratic:
            quadratic_corr = np.round(np.array(quadratic)[:, 2] / precision)
            for edge, _ in enumerate(quadratic):
                i = quadratic[edge][0] + 1
                j = quadratic[edge][1] + 1
                if quadratic[edge][2] > 0:
                    v = np.abs(quadratic_corr[edge])
                    if v != 0:
                        clauses.append([v, -i, -j])
                elif quadratic[edge][2] < 0:
                    v = np.abs(quadratic_corr[edge])
                    if v != 0:
                        clauses.append([v, i, j])
                        clauses.append([v, -i, j])
                        clauses.append([v, i, -j])
                        wcnf_offset -= v
        wcnf_offset = wcnf_offset + bqm.offset / precision
        bqm.variables._relabel(mappings)
        validation_vars = [1, 0, 1, 0, 1, 0, 1, 0]
        validation_weight = 999999
        for v in range(0, len(validation_vars)):
            direction = 1 if validation_vars[v] == 1 else -1
            i = num_variables + 1 + v
            clauses.append([validation_weight, direction * i])
        num_variables += len(validation_vars)
        num_clauses = len(clauses)
        return ConversionResult(
            clauses=clauses,
            num_variables=num_variables,
            num_clauses=num_clauses,
            var_mappings=mappings,
            precision=precision,
            bqm=bqm,
            wcnf_offset=wcnf_offset,
        )
