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

from typing import Tuple

import dimod


def scale_bqm_to_range(
    bqm: dimod.BinaryQuadraticModel,
    *,
    max_abs_coeff: float = 9.0,
) -> Tuple[dimod.BinaryQuadraticModel, float]:
    """
    Scale BQM coefficients to fit within [-max_abs_coeff, +max_abs_coeff].

    Args:
        bqm: BinaryQuadraticModel to scale
        max_abs_coeff: Maximum absolute coefficient value (default: 9.0)

    Returns:
        Tuple of (scaled_bqm, scaling_factor)

    Raises:
        ValueError: If max_abs_coeff is not positive
    """
    if max_abs_coeff <= 0:
        raise ValueError(f"max_abs_coeff must be positive, got {max_abs_coeff}")

    max_abs = 0.0
    for coeff in bqm.linear.values():
        max_abs = max(max_abs, abs(float(coeff)))
    for coeff in bqm.quadratic.values():
        max_abs = max(max_abs, abs(float(coeff)))

    if max_abs == 0.0:
        return bqm.copy(), 1.0

    scaling_factor = max_abs_coeff / max_abs
    scaled_bqm = bqm.copy()
    scaled_bqm.scale(scaling_factor)

    return scaled_bqm, scaling_factor


__all__ = ["scale_bqm_to_range"]
