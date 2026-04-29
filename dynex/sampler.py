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

import logging
import multiprocessing
from typing import Optional

import dimod

from dynex._sampler_core import _DynexSampler
from dynex.config import DynexConfig
from dynex.exceptions import DynexValidationError


class DynexSampler:
    """Dynex neuromorphic computing sampler."""

    def __init__(
        self,
        model,
        logging=True,
        description: Optional[str] = None,
        test=False,
        bnb=True,
        filename_override="",
        config: Optional[DynexConfig] = None,
        preserve_solutions: Optional[bool] = None,
        job_metadata: Optional[dict] = None,
    ):

        # multi-model parallel sampling

        if not config:
            config = DynexConfig()

        self.config = config
        self.logger = config.logger
        self.state = "initialised"
        self.model = model
        self.logging = logging
        self.filename_override = filename_override
        self.description = description if description is not None else config.default_description
        self.test = test
        self.dimod_assignments = {}
        self.bnb = bnb
        self.preserve_solutions = preserve_solutions if preserve_solutions is not None else config.preserve_solutions
        self.use_notebook_output = config.use_notebook_output
        self.job_metadata = job_metadata
        self.timeout = config.default_timeout

    def _log_debug(self, message: str) -> None:
        if self.logger and self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(message)

    @staticmethod
    def _sample_thread(
        q,
        x,
        model,
        logging,
        logger,
        description,
        num_reads,
        annealing_time,
        switchfraction,
        alpha,
        beta,
        gamma,
        delta,
        epsilon,
        zeta,
        minimum_stepsize,
        block_fee,
        shots,
        preserve_solutions,
        qpu_max_coeff,
        config,
    ):
        """Creates a thread for clone sampling."""
        if logging:
            logger.info(f"Clone {x} started...")
        _sampler = _DynexSampler(
            model,
            logging=False,
            description=description,
            test=False,
            config=config,
            preserve_solutions=preserve_solutions,
        )
        _sampleset = _sampler.sample(
            num_reads,
            annealing_time,
            switchfraction,
            alpha,
            beta,
            gamma,
            delta,
            epsilon,
            zeta,
            minimum_stepsize,
            False,
            block_fee,
            shots,
            qpu_max_coeff=qpu_max_coeff,
        )
        if logging:
            logger.info(f"Clone {x} finished")
        q.put(_sampleset)
        return

    def sample(
        self,
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
        rank=1,
        preprocess=False,
        qpu_max_coeff=9.0,
    ):
        """Run sampling and return dimod SampleSet."""

        # assert parameters:
        if clones < 1:
            raise DynexValidationError("Value of clones must be in range [1,128]")
        if clones > 128:
            raise DynexValidationError("Value of clones must be in range [1,128]")
        if not self.config.mainnet and clones > 1:
            raise DynexValidationError("Clone sampling is only supported in network mode")

        # Apollo QPU limitation: annealing_time cannot exceed 10000
        MAX_ANNEALING_TIME_QPU = 10000
        compute_backend = getattr(self.config, "compute_backend", None)
        if compute_backend and (compute_backend == "qpu" or compute_backend == "QPU"):
            if annealing_time > MAX_ANNEALING_TIME_QPU:
                if self.logging:
                    self.logger.warning(
                        f"annealing_time ({annealing_time}) exceeds Apollo QPU limit ({MAX_ANNEALING_TIME_QPU}), "
                        f"capping to {MAX_ANNEALING_TIME_QPU}"
                    )
                annealing_time = MAX_ANNEALING_TIME_QPU

        if clones == 1:
            _sampler = _DynexSampler(
                self.model,
                logging=self.logging,
                description=self.description,
                test=self.test,
                bnb=self.bnb,
                filename_override=self.filename_override,
                config=self.config,
                preserve_solutions=self.preserve_solutions,
                job_metadata=self.job_metadata,
            )
            _sampleset = _sampler.sample(
                num_reads=num_reads,
                annealing_time=annealing_time,
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
                rank=rank,
                preprocess=preprocess,
                qpu_max_coeff=qpu_max_coeff,
            )
            return _sampleset

        else:
            supported_threads = multiprocessing.cpu_count()
            if clones > supported_threads:
                self.logger.info(
                    f"Number of clones > CPU cores: clones: {clones} threads available: {supported_threads}"
                )
            jobs = []
            results = []

            if self.logging:
                self.logger.info(f"Starting sampling ({clones} clones)...")

            # define n samplers:
            for i in range(clones):
                q = multiprocessing.Queue()
                results.append(q)
                p = multiprocessing.Process(
                    target=self._sample_thread,
                    args=(
                        q,
                        i,
                        self.model,
                        self.logging,
                        self.logger,
                        self.description,
                        num_reads,
                        annealing_time,
                        switchfraction,
                        alpha,
                        beta,
                        gamma,
                        delta,
                        epsilon,
                        zeta,
                        minimum_stepsize,
                        block_fee,
                        shots,
                        self.preserve_solutions,
                        qpu_max_coeff,
                        self.config,
                    ),
                )
                jobs.append(p)
                p.start()

            # wait for samplers to finish:
            for job in jobs:
                job.join()

            # collect samples for each job:
            assignments_cum = []
            for result in results:
                assignments = result.get()
                assignments_cum.append(assignments)

            # accumulate and aggregate all results:
            r = None
            for assignment in assignments_cum:
                if len(assignment) > 0:
                    if r is None:
                        r = assignment
                    else:
                        r = dimod.concatenate((r, assignment))

            # aggregate samples:
            r = r.aggregate()

            self.dimod_assignments = r

            return r
