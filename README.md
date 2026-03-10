# Dynex SDK

Python SDK for quantum computing on Dynex platform. Compatible with Dimod, PyQUBO, and Ocean SDK.

## Installation

### With uv

```bash
uv add dynex
```

### Install uv

If you don't have `uv` yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or via Homebrew on macOS:

```bash
brew install uv
```

## Quick Start

```python
import dynex
import dimod

# Create BQM
bqm = dimod.BinaryQuadraticModel({0: 1.0, 1: -1.0}, {(0, 1): 0.5}, 0.0, 'BINARY')

# Create sampler and sample
model = dynex.BQM(bqm)
sampler = dynex.DynexSampler(model)
sampleset = sampler.sample(num_reads=10)

print(sampleset.first.sample)
```

## Configuration

### Environment Variables

Set credentials via environment variables or `.env` file:

```bash
DYNEX_SDK_KEY=your_sdk_key
```

### Compute Backends

Dynex SDK supports three compute backends via the `ComputeBackend` enum:
```python
from dynex import DynexConfig, ComputeBackend

# Using enum (recommended)
config = DynexConfig(compute_backend=ComputeBackend.QPU)

# Using string (also supported)
config = DynexConfig(compute_backend="qpu")
```

#### 1. CPU Backend (Default)

Local CPU simulation for testing and development:

```python
config = DynexConfig(compute_backend=ComputeBackend.CPU)
sampler = dynex.DynexSampler(model, config=config)
sampleset = sampler.sample(num_reads=100)
```

#### 2. GPU Backend

GPU-accelerated simulation:

```python
config = DynexConfig(compute_backend=ComputeBackend.GPU)
sampler = dynex.DynexSampler(model, config=config)
sampleset = sampler.sample(num_reads=100)
```

#### 3. QPU Backend

Access to Quantum Processing Units (QPUs), including Dynex's proprietary quantum hardware (Apollo and Zeus series) and integrated third-party QPUs (IBM, IonQ, Rigetti, D-Wave, QuEra and IQM) via Dynex's Qubit-Agnostic Quantum Platform:

```python
from dynex import QPUModel

# Using QPUModel enum (recommended)
config = DynexConfig(
    compute_backend=ComputeBackend.QPU,
    qpu_model=QPUModel.APOLLO_RC1  # or other supported QPUs
)

# Or using string
config = DynexConfig(
    compute_backend=ComputeBackend.QPU,
    qpu_model="apollo_rc1"  # or other supported QPUs
)

sampler = dynex.DynexSampler(model, config=config)
sampleset = sampler.sample(num_reads=10, annealing_time=500)
```

## Advanced Usage

### BQM Preprocessing

Scale BQM coefficients for QPU compatibility:

```python
import dynex
from dynex import ComputeBackend

# Scale to QPU range
scaled_bqm, scale_factor = dynex.scale_bqm_to_range(bqm, max_abs_coeff=9.0)
model = dynex.BQM(scaled_bqm)

# Sample on QPU
config = DynexConfig(compute_backend=ComputeBackend.QPU, qpu_model=QPUModel.APOLLO_RC1)
sampler = dynex.DynexSampler(model, config=config)
sampleset = sampler.sample(num_reads=10, qpu_max_coeff=9.0)

# Scale energy back
original_energy = sampleset.first.energy / scale_factor
```

### Custom Configuration

Configure SDK behavior through `DynexConfig`:

```python
config = DynexConfig(
    sdk_key="your_sdk_key",
    compute_backend="cpu",
    # Environment parameters (set via config only)
    use_notebook_output=True,      # Dynamic table updates in Jupyter (default: True)
    default_timeout=600.0,          # Max wait time for solutions in seconds (default: 300.0)
    # Job parameters (can override in sampler)
    default_description="My Job",   # Default job description
    preserve_solutions=False,       # Keep solution files after processing (default: False)
)

sampler = dynex.DynexSampler(model, config=config)

# Job parameters can be overridden per sampler
sampler = dynex.DynexSampler(
    model,
    config=config,
    description="Custom Job",        # Override default_description
    preserve_solutions=True,          # Override preserve_solutions
)
```

**Configuration Parameters:**

**Environment Parameters (config only):**
- `use_notebook_output` - Enable dynamic table updates in Jupyter Notebook (default: `True`)
  - `True`: Table updates in-place with `clear_output()` (Jupyter)
  - `False`: Each table printed separately (console/scripts)
- `default_timeout` - Maximum wait time for solutions in seconds (default: `300.0`)

**Job Parameters (config + override):**
- `default_description` - Default job description (default: `"Dynex SDK Job"`)
- `preserve_solutions` - Keep solution files after processing (default: `False`)

### Constrained Quadratic Models (CQM)

```python
from dimod import ConstrainedQuadraticModel, Binary

# Create CQM
cqm = ConstrainedQuadraticModel()
x = [Binary(f'x{i}') for i in range(5)]

# Add objective
cqm.set_objective(sum(x[i] * x[i+1] for i in range(4)))

# Add constraint
cqm.add_constraint(sum(x) == 2)

# Sample
model = dynex.CQM(cqm)
sampler = dynex.DynexSampler(model)
sampleset = sampler.sample(num_reads=10)
```

### Discrete Quadratic Models (DQM)

```python
from dimod import DiscreteQuadraticModel

# Create DQM
dqm = DiscreteQuadraticModel()
dqm.add_variable(3)  # variable with 3 states
dqm.add_variable(2)  # variable with 2 states

# Set interactions
dqm.set_linear(0, [1.0, 2.0, 3.0])
dqm.set_quadratic(0, 1, {(0, 0): 1.0, (1, 1): -1.0})

# Sample
model = dynex.DQM(dqm)
sampler = dynex.DynexSampler(model)
sampleset = sampler.sample(num_reads=10)
```

## Logging & Monitoring

Dynex SDK provides comprehensive logging with platform-specific prefixes and detailed timing information.

### Enable Logging

```python
import logging

logging.basicConfig(level=logging.INFO)

model = dynex.BQM(bqm, logging=True)
sampler = dynex.DynexSampler(model, config=config)
```

### Log Output Features

**Platform Identification:**
```
INFO: [DYNEX-APOLLO-RC1] SAMPLER INITIALISED
INFO: [DYNEX-CPU] SAMPLER INITIALISED
INFO: [DYNEX-GPU] SAMPLER INITIALISED
```

**Problem Settings Summary:**
```
INFO: [DYNEX-APOLLO-RC1] Problem: 400 qubits, 15600 gates
INFO: [DYNEX-APOLLO-RC1] Settings: num_reads=5, shots=3, annealing_time=100
```

**Validation Warnings:**
```
WARNING: [DYNEX-APOLLO-RC1] annealing_time=10 might be short for 400 qubits
WARNING: [DYNEX-APOLLO-RC1] num_reads=100 is very high, consider reducing for faster testing
```

**Shot Progress Tracking:**
```
INFO: [DYNEX-APOLLO-RC1] Shot 1/3 received
INFO: [DYNEX-APOLLO-RC1] Shot 2/3 received
INFO: [DYNEX-APOLLO-RC1] Shot 3/3 received
```

**Timing Breakdown:**
```
INFO: [DYNEX-APOLLO-RC1] Average time per shot: 4.60s (3 shots in 13.80s)
INFO: [DYNEX-APOLLO-RC1] Timing breakdown:
INFO: [DYNEX-APOLLO-RC1]   Job upload:        0.34s
INFO: [DYNEX-APOLLO-RC1]   Time to 1st shot:  7.08s
INFO: [DYNEX-APOLLO-RC1]   Compute (Apollo):  13.80s
INFO: [DYNEX-APOLLO-RC1]   Solution download: 0.15s
INFO: [DYNEX-APOLLO-RC1]   Total elapsed:     14.29s
```

**BQM Scaling Information:**
```
INFO: [DYNEX-APOLLO-RC1] Auto-scaling BQM for QPU: max_abs_coeff=3109.17 > 9.0
INFO: [DYNEX-APOLLO-RC1] BQM scaled by factor 0.002895 for optimal QPU performance
```

### Result Table

The SDK displays a comprehensive result table showing job details:

```
╭────────────┬──────────┬─────────┬─────────────┬─────────┬────────────┬───────────┬───────────┬────────────────╮
│   DYNEXJOB │   QUBITS │   GATES │   NUM_READS │   SHOTS │   ANN.TIME │ ELAPSED   │   WORKERS │   GROUND STATE │
├────────────┼──────────┼─────────┼─────────────┼─────────┼────────────┼───────────┼───────────┼────────────────┤
│       3462 │      400 │   15600 │           5 │       3 │        100 │ 14.29s    │         3 │         179.00 │
╰────────────┴──────────┴─────────┴─────────────┴─────────┴────────────┴───────────┴───────────┴────────────────╯
```

**Columns:**
- `DYNEXJOB` - Job ID on Dynex network
- `QUBITS` - Number of problem variables
- `GATES` - Number of quantum gates/interactions
- `NUM_READS` - Parallel samples per worker
- `SHOTS` - Minimum solutions requested from network
- `ANN.TIME` - Annealing time per sample
- `ELAPSED` - Total elapsed time
- `WORKERS` - Number of workers contributed
- `GROUND STATE` - Best energy found

## Development

Clone the repo and set up the environment with a single command:

```bash
git clone https://github.com/tapok1999/dynex-sdk
cd dynex-sdk
make install
```

This runs `uv sync --group dev` — creates `.venv` and installs all dependencies from the lockfile.

### Running tests

```bash
make test-unit          # unit tests
make test-integration   # integration tests (needs .env with credentials)
make test               # all tests
```

Or directly via uv:

```bash
uv run pytest tests/unit/ -v
```

### Code quality

```bash
make format   # black + isort
make check    # check without changes
make lint     # flake8
```

### Adding dependencies

```bash
uv add <package>              # runtime dependency
uv add --group dev <package>  # dev-only dependency
uv lock                       # regenerate lockfile after manual edits to pyproject.toml
```

### Building

```bash
make build    # produces dist/dynex-*.whl and dist/dynex-*.tar.gz
```

## License

BSD-3-Clause
