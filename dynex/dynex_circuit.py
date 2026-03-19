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
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES LOSS OF USE, DATA, OR PROFITS OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import base64
import inspect
import json
import secrets
import warnings
import zlib
from collections import Counter
from dataclasses import dataclass

import pennylane as qml
from pennylane import numpy as np

import dynex
from dynex import DynexConfig
from dynex.exceptions import DynexJobError, DynexValidationError

warnings.filterwarnings("ignore", category=DeprecationWarning)


@dataclass(frozen=True)
class CircuitEncoding:
    data: str
    is_qpe: bool
    is_grover: bool
    is_cqu: bool
    is_qu: bool


class CircuitModel:
    """
    `Internal Class` to hold information about the Dynex circuit
    """

    def __init__(self, circuit_str=None, wires=None, params=None):
        self.qasm_circuit = None
        self.circuit_str = circuit_str
        self.qasm_filepath = "tmp/"
        self.qasm_filename = secrets.token_hex(16) + ".qasm.dnx"
        self.params = params
        self.wires = wires
        self.type = "qasm"
        self.type_str = "QASM"
        self.bqm = None
        self.clauses = []
        self.wcnf_offset = 0
        self.precision = 1.0


class DynexCircuit:
    description: str = "Dynex SDK Job"

    def __init__(self, config: DynexConfig = None):
        self.config = config if config is not None else DynexConfig()
        self.logger = self.config.logger

    def sol2state(self, sample, wires, is_qpe, is_grover, is_cqu, is_qu):
        state = [0] * wires
        # Check if sample is a dict with WCNF variable indices (QASM converted to WCNF)
        if isinstance(sample, dict):
            # Try to find qubit keys first (original QASM format)
            has_qubit_keys = any(f"q_{i}_real" in sample or f"q_{i}_imag" in sample for i in range(wires))

            if not has_qubit_keys:
                # This is likely a WCNF solution - sample contains WCNF variable indices
                # For WCNF, sample keys are variable indices (0, 1, 2, ...) and values are 0 or 1
                # We need to map WCNF variables back to qubit wires
                # But we don't have var_mappings here, so try direct mapping
                # Assume sequential mapping: WCNF variable i -> qubit wire i (for first wires)
                for wire in range(wires):
                    if wire in sample:
                        state[wire] = sample[wire]
                    elif isinstance(sample, dict) and len(sample) > wire:
                        # Try to get value by position if keys are sequential
                        values = list(sample.values())
                        if wire < len(values):
                            state[wire] = values[wire]
                return state

        # Original logic for QASM with qubit keys
        for wire in range(wires):
            r_key = f"q_{wire}_real"
            i_key = f"q_{wire}_imag"
            qpe_key = f"q_{wire}_ctrl_qpe_imag"

            # Check if keys exist (sample is SampleView, might throw ValueError on missing key)
            try:
                if is_qpe and qpe_key in sample:
                    state[wire] = 1 if sample[qpe_key] > sample[r_key] else 0
                elif r_key in sample and i_key in sample:
                    if is_grover or is_cqu or is_qu:
                        state[wire] = 1 if sample[i_key] > 0.5 else 0
                    else:
                        state[wire] = 1 if sample[r_key] > 0.5 else 0
                else:
                    raise KeyError("Keys not found")
            except (KeyError, ValueError):
                # Try to find alternative gate-specific keys (e.g., for last qubit)
                # Common gate suffixes in order of preference
                gate_suffixes = ["h", "rz", "cp", "swap", "basis"]
                found = False
                for suffix in gate_suffixes:
                    r_key_alt = f"q_{wire}_{suffix}_real"
                    i_key_alt = f"q_{wire}_{suffix}_imag"
                    try:
                        if r_key_alt in sample and i_key_alt in sample:
                            if is_grover or is_cqu or is_qu:
                                state[wire] = 1 if sample[i_key_alt] > 0.5 else 0
                            else:
                                state[wire] = 1 if sample[r_key_alt] > 0.5 else 0
                            found = True
                            break
                    except (KeyError, ValueError):
                        continue
                if not found:
                    self.logger.info(f"Warning: No final state found for wire {wire}")
        return state

    def get_samples(self, sampleset, wires, is_qpe, is_grover, is_cqu, is_qu):
        samples = []
        for solution, occurrence in zip(sampleset, sampleset.record.num_occurrences):
            sample = self.sol2state(solution, wires, is_qpe, is_grover, is_cqu, is_qu)
            samples.extend([sample] * occurrence)
        return samples

    def get_probs(self, sampleset, wires, is_qpe, is_grover, is_cqu, is_qu):
        state_counts = Counter()
        total_samples = sum(sampleset.record.num_occurrences)
        for solution, occurrence in zip(sampleset, sampleset.record.num_occurrences):
            state = self.sol2state(solution, wires, is_qpe, is_grover, is_cqu, is_qu)
            state_counts[tuple(state)] += occurrence
        qubit_probs = np.zeros(wires)
        for state, count in state_counts.items():
            for i, bit in enumerate(state):
                if bit == 1:
                    qubit_probs[i] += count / total_samples
        return qubit_probs[::-1]

    @staticmethod
    def _save_qasm_file(dnx_circuit):
        """
        `Internal Function`

        Saves the circuit as a .qasm file locally in /tmp as defined in dynex.ini
        """

        filename = dnx_circuit.qasm_filepath + dnx_circuit.qasm_filename

        with open(filename, "w", encoding="utf-8") as f:
            f.write(dnx_circuit.circuit_str)

    @staticmethod
    def check_pennylane_circuit(circuit) -> bool:
        if isinstance(circuit, qml.QNode):
            return True
        if hasattr(circuit, "quantum_instance") and isinstance(circuit.quantum_instance, qml.QNode):
            return True
        if inspect.isfunction(circuit):
            source = inspect.getsource(circuit)
            pops = [
                "qml.Hadamard",
                "qml.CNOT",
                "qml.RX",
                "qml.RY",
                "qml.RZ",
                "qml.BasisEmbedding",
                "qml.QFT",
                "qml.adjoint",
                "qml.state",
                "qml.sample",
                "qml.PauliX",
                "qml.PauliY",
                "qml.PauliZ",
                "qml.S",
                "qml.T",
                "qml.CZ",
                "qml.SWAP",
                "qml.CSWAP",
                "qml.Toffoli",
                "qml.PhaseShift",
                "qml.ControlledPhaseShift",
                "qml.CRX",
                "qml.CRY",
                "qml.CRZ",
                "qml.Rot",
                "qml.MultiRZ",
                "qml.QubitUnitary",
                "qml.ControlledQubitUnitary",
                "qml.IsingXX",
                "qml.IsingYY",
                "qml.IsingZZ",
                "qml.Identity",
                "qml.Kerr",
                "qml.CrossKerr",
                "qml.Squeezing",
                "qml.DisplacedSqueezed",
                "qml.TwoModeSqueezing",
                "qml.ControlledAddition",
                "qml.ControlledSubtraction",
            ]
            if "qml." in source and any(op in source for op in pops):
                return True
            if "wires=" in source:
                return True
        if hasattr(circuit, "interface") or hasattr(circuit, "device"):
            return True
        if hasattr(circuit, "func") and hasattr(circuit, "device"):
            return True
        return False

    @staticmethod
    def _qiskit_to_circuit(qc, circuit_params, wires):
        _wires = list(range(wires))
        my_qfunc = qml.from_qiskit(qc)

        def pl_circuit(params):
            my_qfunc(wires=_wires)

        return pl_circuit

    @staticmethod
    def _qasm_to_circuit(t, circuit_params, wires):
        """
        `Internal Function`

        Reads raw qasm text and converts to PennyLane Circuit class object
        """
        _wires = list(range(wires))
        qasm_circuit = qml.from_qasm(t, measurements=[])

        # Return a plain function (not a QNode) so that _pennylane_to_file can
        # capture its operations via the QuantumTape context manager.
        # QNodes execute eagerly and don't propagate ops to an outer tape.
        def pl_circuit(params):
            qasm_circuit(wires=_wires)

        return pl_circuit

    @staticmethod
    def _pennylane_to_file(circuit, params, wires) -> CircuitEncoding:
        with qml.tape.QuantumTape() as tape:
            circuit(params)
        ops = tape.operations
        is_qpe = any(op.name.startswith("QuantumPhaseEstimation") for op in ops)
        is_grover = any(op.name.startswith("GroverOperator") for op in ops)
        is_cqu = any(op.name.startswith("ControlledQubitUnitary") for op in ops)
        is_qu = any(op.name.startswith("QubitUnitary") for op in ops)

        def process_ops(op):
            op_dict = {
                "name": op.name,
                "wires": [int(w) for w in op.wires],  # ensure wires are integers
                "params": [p.tolist() if hasattr(p, "tolist") else p for p in op.parameters],
                "hyperparams": {
                    k: v.tolist() if hasattr(v, "tolist") else v for k, v in op.hyperparameters.items() if k != "wires"
                },  # For B.E gate
                "adjointD": 0,  # supporting nested daggers
                "ctrlD": 0,  # supporting nested controlled gates
            }
            name = op.name
            if name.startswith("Snapshot"):
                pass
            while name.startswith(("Adjoint(", "C(")):
                if name.startswith("Adjoint("):
                    op_dict["adjointD"] += 1
                    name = name[8:-1]  # remove "Adjoint(" and ")"
                elif name.startswith("C("):
                    op_dict["ctrlD"] += 1
                    name = name[2:-1]  # remove "C(" and ")"
            op_dict["base_name"] = name
            if op_dict["ctrlD"] > 0 or name == "ControlledQubitUnitary":  # handling CQU
                op_dict["control_wires"] = [int(w) for w in op.control_wires]
                op_dict["target_wires"] = [int(w) for w in op.wires[len(op.control_wires) :]]
            if name == "QuantumPhaseEstimation":  # handling QPE
                op_dict["estimation_wires"] = [int(w) for w in op.hyperparameters["estimation_wires"]]
                U = op.hyperparameters["unitary"]
                if isinstance(U, qml.operation.Operation):
                    op_dict["unitary"] = {
                        "name": U.name,
                        "wires": [int(w) for w in U.wires],
                        "params": [p.tolist() if hasattr(p, "tolist") else p for p in U.parameters],
                    }
                else:
                    op_dict["unitary"] = U.tolist() if hasattr(U, "tolist") else U
                    op_dict["target_wires"] = [int(w) for w in op.target_wires]
            return op_dict

        cir_info = [process_ops(op) for op in ops]
        cir_i = {"operations": cir_info, "nWires": wires, "nParams": len(params), "params": params}
        data = json.dumps(cir_i, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
        comp = zlib.compress(data.encode("utf-8"))
        dynex_circuit = base64.b85encode(comp).decode("utf-8")
        return CircuitEncoding(data=dynex_circuit, is_qpe=is_qpe, is_grover=is_grover, is_cqu=is_cqu, is_qu=is_qu)

    def execute(
        self,
        circuit,
        params,
        wires,
        num_reads=1000,
        integration_steps=100,
        method="measure",
        logging=True,
        bnb=False,
        switchfraction=0.0,
        alpha=20,
        beta=20,
        gamma=1,
        delta=1,
        epsilon=1,
        zeta=1,
        minimum_stepsize=0.05,
        block_fee=0,
        shots=1,
        qpu_max_coeff=9.0,
    ):
        """
        Function to execute quantum gate based circuits natively on the Dynex Neuromorphic Computing Platform.

        :Parameters:
        - :circuit: A circuit in one of the following formats: [openQASM, PennyLane, Qiskit, Cirq] (circuit class)
        - :params: Parameters for circuit execution (`list`)
        - :wires: number of qubits (`int`)
        - :method: Type of circuit measurement:
            'measure': samples of a single measurement
            'probs': computational basis state probabilities
            'all': all solutions as arrays
            'sampleset': dimod sampleset
        - :shots: Sets the minimum number of solutions to retrieve from the network. Works both on local mode and network mode (Default: 1). Typically used for situations where not only the best global optimum (sampleset.first) is required, but multiple optima from different workers (`int`).
        - :description: Defines the description for the job, which is shown in Dynex job dashboards as well as in the network explorer (`string`)

        :Sampling Parameters:

        - :num_reads: Defines the number of parallel samples to be performed (`int` value in the range of [32, MAX_NUM_READS] as defined in your license)

        - :annealing_time: Defines the number of integration steps for the sampling. Models are being converted into neuromorphic circuits, which are then simulated with ODE integration by the participating workers (`int` value in the range of [1, MAX_ANNEALING_TIME] as defined in your license)

        - :switchfraction: Defines the percentage of variables which are replaced by random values during warm start samplings (`double` in the range of [0.0, 1.0])


        - :debugging: Only applicable for test-net sampling. Defines if the sampling process should be quiet with no terminal output (FALSE) or if process updates are to be shown (TRUE) (`bool`)

        - :bnb: Use alternative branch-and-bound sampling when in local mode (`bool`)

        - :block_fee: Computing jobs on the Dynex platform are being prioritised by the block fee which is being offered for computation. If this parameter is not specified, the current average block fee on the platform is being charged. To set an individual block fee for the sampling, specify this parameter, which is the amount of DNX in nanoDNX (1 DNX = 1,000,000,000 nanoDNX)

        :Returns:

        - Returns the measurement based on the parameter 'measure'

        :Example:

        .. code-block:: Python

            params = [0.1, 0.2]
            wires = 2

            dev = qml.device('default.qubit', wires=wires, shots=1)
            @qml.qnode(dev)
            def circuit(params):
                qml.RX(params[0], wires=0)
                qml.RY(params[1], wires=1)
                qml.CNOT(wires=[0, 1])
                qml.Hadamard(wires=0)
                return qml.sample()#qml.expval(qml.Hadamard(0)) #qml.sample() #state()

            # Draw circuit:
            _ = qml.draw_mpl(circuit, style="black_white", expansion_strategy="device")(params)

            # Compute circuit on Dynex:
            import dynex_circuit
            measure = dynex_circuit.execute(circuit, params, network mode)
            print(measure)

            │   DYNEXJOB │   QUBITS │   QUANTUM GATES │   BLOCK FEE │   ELAPSED │   WORKERS READ │   CIRCUITS │   STEPS │   GROUND STATE │
            ├────────────┼──────────┼─────────────────┼─────────────┼───────────┼────────────────┼────────────┼─────────┼────────────────┤
            │      28391 │       21 │              64 │        0.00 │      0.58 │              1 │       1000 │     256 │       38708.00 │
            ╰────────────┴──────────┴─────────────────┴─────────────┴───────────┴────────────────┴────────────┴─────────┴────────────────╯
            ╭────────────┬─────────────────┬────────────┬───────┬──────────┬──────────────┬─────────────────────────────┬───────────┬──────────╮
            │     WORKER │         VERSION │   CIRCUITS │   LOC │   ENERGY │      RUNTIME │                 LAST UPDATE │     STEPS │   STATUS │
            ├────────────┼─────────────────┼────────────┼───────┼──────────┼──────────────┼─────────────────────────────┼───────────┼──────────┤
            │ 1147..9be1 │ 2.3.5.OZM.134.L │       1000 │     0 │     0.00 │ 4.190416548s │ 2024-08-06T19:37:36.148518Z │ 0 (0.00%) │  STOPPED │
            ├────────────┼─────────────────┼────────────┼───────┼──────────┼──────────────┼─────────────────────────────┼───────────┼──────────┤
            │ 6a66..2857 │ 2.3.5.OZM.134.L │       1000 │     0 │     0.00 │ 9.002006172s │  2024-08-06T19:37:31.33693Z │ 0 (0.00%) │  STOPPED │
            ╰────────────┴─────────────────┴────────────┴───────┴──────────┴──────────────┴─────────────────────────────┴───────────┴──────────╯
            FINISHED READ AFTER 57.94 SECONDS
            SAMPLESET READY
            [1 0]

        ...
        """

        # enforce param wires to int value:
        if isinstance(wires, list):
            wires = len(wires)
        enc: CircuitEncoding | None = None
        if self.check_pennylane_circuit(circuit):
            enc = self._pennylane_to_file(circuit, params, wires)
            if logging:
                self.logger.info("Executing PennyLane quantum circuit")

        qasm_checker = isinstance(circuit, str)
        if qasm_checker:
            circuit = self._qasm_to_circuit(circuit, params, wires)
            enc = self._pennylane_to_file(circuit, params, wires)
            if logging:
                self.logger.info("Executing OpenQASM quantum circuit")

        qiskit_checker = str(type(circuit)).find("qiskit") > 0
        if qiskit_checker:
            circuit = self._qiskit_to_circuit(circuit, params, wires)
            enc = self._pennylane_to_file(circuit, params, wires)
            if logging:
                self.logger.info("Executing Qiskit quantum circuit")

        circuit_str = enc.data if enc else None
        is_qpe = enc.is_qpe if enc else False
        is_grover = enc.is_grover if enc else False
        is_cqu = enc.is_cqu if enc else False
        is_qu = enc.is_qu if enc else False

        circ_model = CircuitModel(circuit_str=circuit_str, params=params, wires=wires)

        self._save_qasm_file(circ_model)

        # Automatically set job_metadata for Circuit BQM (QASM)
        job_metadata = {"type": "qasm"}

        sampler = dynex.DynexSampler(
            model=circ_model,
            description=self.description,
            bnb=bnb,
            logging=logging,
            filename_override="",  # Don't override filename for QASM - let SDK generate a .dnx file for results
            config=self.config,
            job_metadata=job_metadata,
        )  # Automatically identify Circuit BQM

        sampleset = sampler.sample(
            num_reads=num_reads,
            annealing_time=integration_steps,
            switchfraction=switchfraction,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            epsilon=epsilon,
            zeta=zeta,
            minimum_stepsize=minimum_stepsize,
            debugging=logging,
            block_fee=block_fee,
            shots=shots,
            qpu_max_coeff=qpu_max_coeff,
        )

        # decode solution:
        if method not in ["measure", "probs", "all", "sampleset"]:
            raise DynexValidationError("Method must be either 'measure', 'probs', 'all' or 'sampleset'")

        if logging:
            self.logger.info(f"-------------- /  {method}  / ------------")
        if method == "measure":
            # Check if sampleset is empty
            if len(sampleset) == 0:
                self.logger.error(
                    "No valid solution found. The quantum circuit may be too complex or the solver timed out."
                )
                raise DynexJobError("SampleSet is empty - no solution was found for the quantum circuit")

            # For QASM converted to WCNF, we need to use var_mappings to decode
            # Check if we have access to var_mappings through the sampler
            first_sample = sampleset.first.sample
            state = None

            # Try to get var_mappings from the internal sampler
            if hasattr(sampler, "_sampler") and hasattr(sampler._sampler, "model"):
                model = sampler._sampler.model
                if hasattr(model, "var_mappings") and model.var_mappings:
                    # var_mappings maps qubit wire -> WCNF variable
                    # But in dimod SampleSet, keys are the original variable names (qubit wires)
                    # So first_sample should already have qubit wire indices as keys
                    # Extract qubit states from WCNF solution
                    state = [0] * wires
                    if isinstance(first_sample, dict):
                        # For QASM converted to WCNF, var_mappings maps qubit wire -> WCNF variable index
                        # But dimod SampleSet uses original variable names (qubit wires) as keys
                        # So we can directly use wire indices from var_mappings
                        for wire_idx in range(wires):
                            # Try to find the value for this wire
                            if wire_idx in first_sample:
                                # Direct access by wire index
                                state[wire_idx] = first_sample[wire_idx]
                            elif wire_idx in model.var_mappings:
                                # Try using the WCNF variable from var_mappings
                                wcnf_var = model.var_mappings[wire_idx]
                                if wcnf_var in first_sample:
                                    state[wire_idx] = first_sample[wcnf_var]
                            # Also try sequential access (WCNF variables are relabeled to 0, 1, 2, ...)
                            # If var_mappings uses sequential indices, try direct index
                            if state[wire_idx] == 0 and wire_idx < len(first_sample):
                                # Try to get by position if keys are sequential
                                sample_keys = sorted([k for k in first_sample.keys() if isinstance(k, int)])
                                if wire_idx < len(sample_keys):
                                    state[wire_idx] = first_sample[sample_keys[wire_idx]]

            # If we couldn't decode using var_mappings, try original method
            if state is None:
                samples = self.get_samples(sampleset, wires, is_qpe, is_grover, is_cqu, is_qu)
                if samples and len(samples) > 0:
                    state = samples[0]
                else:
                    # Last resort: try to extract directly
                    if logging:
                        self.logger.warning(
                            "Could not decode using var_mappings or get_samples, trying direct extraction"
                        )
                    if isinstance(first_sample, dict):
                        # Extract values in order of variable indices
                        state = [first_sample.get(i, 0) for i in range(min(wires, len(first_sample)))]
                        # Pad with zeros if needed
                        while len(state) < wires:
                            state.append(0)
                    else:
                        state = [0] * wires

            if is_qpe:
                result = np.array(state)
            else:
                result = np.array(state)[::-1]
        elif method == "sampleset":
            result = sampleset
        elif method == "all":
            result = [
                np.array(sample) for sample in self.get_samples(sampleset, wires, is_qpe, is_grover, is_cqu, is_qu)
            ]
        else:  # probs
            probs = self.get_probs(sampleset, wires, is_qpe, is_grover, is_cqu, is_qu)
            result = probs
        return result
