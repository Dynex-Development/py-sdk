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

import dimod

from dynex.config import DynexConfig

from .base import DynexModel


class BQM(DynexModel):
    """Binary Quadratic Model for Dynex neuromorphic computing."""

    def __init__(self, bqm, relabel=True, logging=False, formula=2, config: DynexConfig = None):
        super().__init__(config=config, logging=logging)

        if isinstance(bqm, dict):
            bqm = dimod.BinaryQuadraticModel.from_qubo(bqm)

        if formula == 1:
            r = self._convert_bqm_to_qubo(bqm, logging)
        elif formula == 2:
            r = self._convert_bqm_to_qubo_direct(bqm, logging)
        else:
            raise Exception(f"Unknown value of formula: {formula}. It must be in [1, 2].")
        self.clauses = r.clauses
        self.num_variables = r.num_variables
        self.num_clauses = r.num_clauses
        self.var_mappings = r.var_mappings
        self.precision = r.precision
        self.bqm = r.bqm
        self.wcnf_offset = r.wcnf_offset
        if (self.num_clauses + self.num_variables) == 0:
            raise Exception("Could not initiate model - no variables & clauses.")
        self.type = "wcnf"
        self.logging = logging
        self.type_str = "BQM"

    def __str__(self) -> str:
        return self.type_str
