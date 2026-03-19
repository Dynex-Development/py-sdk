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

import warnings
from importlib.metadata import version

# Enforce mandatory dependency on zstandard: SDK will not run without it
try:
    import zstandard as _zstd  # type: ignore  # noqa: F401
except Exception as e:
    raise ImportError(
        "Dynex SDK requires the 'zstandard' package to be installed and importable. "
        "Please install it with: pip install 'zstandard>=0.22'"
    ) from e

from .api import DynexAPI
from .compute_backend import ComputeBackend
from .config import DynexConfig
from .dynex_circuit import DynexCircuit
from .exceptions import (
    DynexConnectionError,
    DynexError,
    DynexJobError,
    DynexModelError,
    DynexSolverError,
    DynexValidationError,
)
from .models import BQM, CQM, DQM
from .preprocessing import scale_bqm_to_range
from .qpu_models import QPUModel
from .sampler import DynexSampler
from .utils import sample_qubo

__author__ = "Dynex Developers"
__credits__ = "Dynex Developers, Contributors, Supporters and the Dynex Community"

try:
    __version__ = version("dynex")
except Exception:
    __version__ = "0.0.0-dev"
