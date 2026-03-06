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

import hashlib

import dimod
import numpy as np

from dynex import BQM, DynexConfig, DynexSampler


def calculate_sha3_256_hash(string):
    """Calculate SHA3-256 hash of string."""
    sha3_256_hash = hashlib.sha3_256()
    sha3_256_hash.update(string.encode("utf-8"))
    return sha3_256_hash.hexdigest()


def calculate_sha3_256_hash_bin(bin):
    """Calculate SHA3-256 hash of binary data."""
    sha3_256_hash = hashlib.sha3_256()
    sha3_256_hash.update(bin)
    return sha3_256_hash.hexdigest()


def max_value(inputlist):
    """Get maximum value from list of sublists."""
    return max([sublist[-1] for sublist in inputlist])


def sample_qubo(
    Q,
    offset=0.0,
    logging=True,
    formula=2,
    description="Dynex SDK Job",
    bnb=True,
    num_reads=32,
    annealing_time=10,
    clones=1,
    switchfraction=0.0,
    alpha=20,
    beta=20,
    gamma=1,
    delta=1,
    epsilon=1,
    zeta=1,
    minimum_stepsize=0.05,
    debugging=False,
    block_fee=0,
    shots=1,
    config=None,
):
    """Sample a QUBO problem using Dynex neuromorphic computing."""
    if config is None:
        config = DynexConfig()

    if isinstance(Q, np.ndarray):
        Q = {(i, j): Q[i, j] for i in range(Q.shape[0]) for j in range(i, Q.shape[1]) if Q[i, j] != 0}

    bqm = dimod.BinaryQuadraticModel.from_qubo(Q, offset)
    model = BQM(bqm, logging=logging, formula=formula, config=config)
    sampler = DynexSampler(model, logging=logging, description=description, bnb=bnb, config=config)
    sampleset = sampler.sample(
        num_reads=num_reads,
        annealing_time=annealing_time,
        clones=clones,
        switchfraction=switchfraction,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        epsilon=epsilon,
        zeta=zeta,
        minimum_stepsize=minimum_stepsize,
        debugging=debugging,
        block_fee=block_fee,
        shots=shots,
    )
    return sampleset
