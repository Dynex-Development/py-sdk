# Migration Guide: Dynex SDK Legacy to V2

This guide covers the migration from the legacy Dynex SDK (v0.1.x) to SDK V2. The new version introduces significant architectural changes, improved configuration management, and a modern gRPC-based communication protocol.

## Overview of Changes

SDK V2 represents a complete rewrite with the following major changes:

- **Communication Protocol**: Migrated from HTTP/FTP to gRPC
- **Configuration System**: Replaced `dynex.ini` with environment variables and `.env` files
- **Backend Architecture**: New QuantumRouterEngine backend replaces legacy HTTP API
- **Code Organization**: Modular structure instead of monolithic implementation
- **Compute Backends**: Introduced explicit backend selection (CPU, GPU, QPU)
- **Dependencies**: Removed FTP libraries, added gRPC and Protocol Buffers

## Configuration Migration

### Legacy Configuration (dynex.ini)

The old SDK used `dynex.ini` for configuration:

```ini
[DYNEX]
API_ENDPOINT = https://api.dynex.com
API_KEY = your_api_key
API_SECRET = your_api_secret

[FTP_COMPUTING_FILES]
ftp_hostname = ftp.dynex.com
ftp_username = user
ftp_password = pass
ftp_path = /compute/
downloadurl = https://files.dynex.com

[FTP_SOLUTION_FILES]
ftp_hostname = ftp.dynex.com
ftp_username = user
ftp_password = pass
```

### New Configuration (Environment Variables)

SDK V2 uses environment variables with optional `.env` file support:

**Create `.env` file:**

```bash
DYNEX_SDK_KEY=your_sdk_key
DYNEX_GRPC_ENDPOINT=127.0.0.1:9090
```

**Or set environment variables directly:**

```bash
export DYNEX_SDK_KEY=your_sdk_key
export DYNEX_GRPC_ENDPOINT=127.0.0.1:9090
```

**In code:**

```python
from dynex import DynexConfig, ComputeBackend

# CPU/GPU backend - network mode
config = DynexConfig(
    compute_backend=ComputeBackend.CPU  # or GPU
)

# LOCAL backend - local simulation (testnet)
config = DynexConfig(
    compute_backend=ComputeBackend.LOCAL
)

# QPU backend - quantum hardware (network mode)
config = DynexConfig(
    sdk_key="your_sdk_key",
    grpc_endpoint="grpc.dynex.com:9090",
    compute_backend=ComputeBackend.QPU,
    qpu_model=QPUModel.APOLLO_RC1  # Required for QPU (or use string "apollo_rc1")
)
```

### Configuration Priority

SDK V2 uses the following priority order:

1. Function parameters (highest)
2. Environment variables (`DYNEX_*`)
3. `.env` file (if python-dotenv installed)
4. Default values (lowest)

## API Changes

### Sampler Initialization

**Legacy:**

```python
import dynex

model = dynex.BQM(bqm)
sampler = dynex.DynexSampler(
    model,
    logging=True,
    description="My Job"
)
```

**SDK V2:**

```python
import dynex
from dynex import DynexConfig, ComputeBackend

# For CPU/GPU backends
config = DynexConfig(
    compute_backend=ComputeBackend.CPU  # or GPU
)

# For QPU backend - qpu_model is required in config
config_qpu = DynexConfig(
    compute_backend=ComputeBackend.QPU,
    qpu_model=QPUModel.APOLLO_RC1  # Required for QPU (or use string "apollo_rc1")
)

model = dynex.BQM(bqm)
sampler = dynex.DynexSampler(
    model,
    config=config_qpu,
    logging=True,
    description="My Job"
)
```

### Sampling Methods

**Legacy:**

```python
# Basic sampling
sampleset = sampler.sample(num_reads=100, annealing_time=10)

# Version selection
sampleset = sampler.sample(num_reads=100, v2=True)
```

**SDK V2:**

```python
# CPU/GPU backend (same interface)
config = DynexConfig(
    compute_backend=ComputeBackend.CPU  # or GPU
)
sampler = dynex.DynexSampler(model, config=config)
sampleset = sampler.sample(num_reads=100)

# QPU backend - qpu_model must be specified in config
from dynex import QPUModel

config = DynexConfig(
    compute_backend=ComputeBackend.QPU,
    qpu_model=QPUModel.APOLLO_RC1  # Required for QPU
)
sampler = dynex.DynexSampler(model, config=config)
sampleset = sampler.sample(
    num_reads=10,
    annealing_time=500
    # qpu_model is no longer a parameter here - it's in config
)
```

### Model Types

Model initialization remains compatible:

```python
# BQM - no changes
from dimod import BinaryQuadraticModel
bqm = BinaryQuadraticModel({0: 1.0}, {(0, 1): -1.0}, 0.0, 'BINARY')
model = dynex.BQM(bqm)

# CQM - no changes
from dimod import ConstrainedQuadraticModel
cqm = ConstrainedQuadraticModel()
model = dynex.CQM(cqm)

# DQM - no changes
from dimod import DiscreteQuadraticModel
dqm = DiscreteQuadraticModel()
model = dynex.DQM(dqm)

```

## Compute Backend Selection

SDK V2 introduces explicit backend selection through the `ComputeBackend` enum.

### Available Backends

```python
from dynex import ComputeBackend

# CPU/GPU backend - network mode
config = DynexConfig(compute_backend=ComputeBackend.CPU)
config = DynexConfig(compute_backend=ComputeBackend.GPU)

# LOCAL backend - local simulation (testnet)
config = DynexConfig(compute_backend=ComputeBackend.LOCAL)

# QPU backend - quantum hardware (network mode)
config = DynexConfig(compute_backend=ComputeBackend.QPU, qpu_model=QPUModel.APOLLO_RC1)

# Unspecified - auto-select (default: CPU)
config = DynexConfig(compute_backend=ComputeBackend.UNSPECIFIED)
```

### QPU Models

For QPU backend, you must specify a QPU model using the `QPUModel` enum:

```python
from dynex import QPUModel

# Using enum (recommended)
config = DynexConfig(
    compute_backend=ComputeBackend.QPU,
    qpu_model=QPUModel.APOLLO_RC1
)

# Or use another model
config = DynexConfig(
    compute_backend=ComputeBackend.QPU,
    qpu_model=QPUModel.APOLLO_10000
)

# Using string (also supported)
config = DynexConfig(
    compute_backend=ComputeBackend.QPU,
    qpu_model="apollo_rc1"  # or "apollo_10000"
)
```

**Available QPU Models:**
- `QPUModel.APOLLO_RC1` → `"apollo_rc1"` (Apollo RC1 chip)
- `QPUModel.APOLLO_10000` → `"apollo_10000"` (Apollo 10000 chip)

### Backend-Specific Parameters

**CPU/GPU Backends (Network Mode):**
```python
from dynex import ComputeBackend

# Network mode - CPU/GPU backends
config = DynexConfig(
    compute_backend=ComputeBackend.CPU  # or ComputeBackend.GPU
)
sampler = dynex.DynexSampler(model, config=config)
sampleset = sampler.sample(
    num_reads=100,  # GPU can handle more: 1000+
    annealing_time=10
)
```

**LOCAL Backend (Local Simulation):**
```python
from dynex import ComputeBackend

# Local simulation - testnet mode (replaces )
config = DynexConfig(
    compute_backend=ComputeBackend.LOCAL
)
sampler = dynex.DynexSampler(model, config=config)
sampleset = sampler.sample(
    num_reads=100,
    annealing_time=10
)
```

**QPU Backend:**
```python
from dynex import ComputeBackend, QPUModel

# qpu_model must be specified in config (required for QPU)
config = DynexConfig(
    compute_backend=ComputeBackend.QPU,
    qpu_model=QPUModel.APOLLO_RC1  # or QPUModel.APOLLO_10000 or "apollo_rc1"
)
sampler = dynex.DynexSampler(model, config=config)
sampleset = sampler.sample(
    num_reads=10,
    annealing_time=500,
    preprocess=True
)
```

**Important:** 
- `qpu_model` is required in `DynexConfig` when using QPU backend
- `qpu_model` cannot be passed to `sample()` - must be in config
- Backend determines mode: LOCAL (offline), others (network)

## Protocol Changes

### Legacy: HTTP + FTP

The old SDK used:
- HTTP REST API for job management
- FTP for uploading compute files
- FTP for downloading solution files
- Polling mechanism for solution retrieval

### SDK V2: gRPC

The new SDK uses:
- gRPC for all communication
- Protocol Buffers for data serialization
- Streaming for real-time solution updates
- No FTP dependencies

**Migration Impact:**

- Firewall rules must allow gRPC traffic (default port 9090)
- FTP server access no longer required
- Faster communication and lower latency
- Binary protocol instead of text-based

## Dependencies

### Removed Dependencies

The following are no longer required:

- `ftplib` - FTP client removed
- HTTP client libraries for legacy API
- File polling mechanisms

### New Dependencies

Add these to your environment:

```bash
pip install grpcio>=1.60.0
pip install grpcio-tools>=1.60.0
pip install zstandard>=0.22.0
```

**Full installation:**

```bash
pip install dynex
```

Or with development dependencies:

```bash
pip install dynex[dev]
```

## Code Structure Changes

### Legacy Structure

```
dynex/
  __init__.py          # ~3000+ lines monolithic file
  dynex_circuit.py
  dynex.ini           # Configuration file
```

### SDK V2 Structure

```
dynex/
  __init__.py         # Clean imports only
  api.py             # API interface layer
  config.py          # Configuration management
  sampler.py         # Sampling logic
  models/            # Model definitions (BQM, CQM, DQM, SAT)
    __init__.py      # Model exports
    base.py          # Base model class
    bqm.py           # Binary Quadratic Model
    cqm.py           # Constrained Quadratic Model
    dqm.py           # Discrete Quadratic Model
  preprocessing.py   # BQM preprocessing utilities
  compute_backend.py # Backend enum
  qpu_models.py      # QPU model definitions
  grpc_client.py     # gRPC communication
  utils.py           # Utility functions
  proto/             # Protocol Buffer definitions
    sdk_pb2.py
    sdk_pb2_grpc.py
  interfaces/        # Abstract interfaces
    api.py
```

## Feature Compatibility

### Maintained Features

- BQM, CQM, DQM model support
- Dimod integration
- PyQUBO compatibility
- Network and local modes via backend selection
- Annealing time configuration
- Multiple reads support

### Changed Features

- **Backend selection**: Now explicit via `ComputeBackend` enum
- **Configuration**: Environment variables instead of INI files
- **Communication**: gRPC instead of HTTP/FTP
- **QPU models**: Explicit selection via `QPUModel` enum
- **Version selection**: Removed (v2 is default)

### Removed Features

- FTP-based file transfer
- Legacy v1 mode
- Progress bars during FTP upload/download
- `dynex.ini` configuration file
- HTTP REST API calls
- Branch-and-bound (`bnb`) parameter for testnet

## Migration Steps

### Step 1: Update Installation

```bash
# Uninstall old SDK
pip uninstall dynex

# Install new SDK
pip install dynex
```

### Step 2: Remove dynex.ini

Delete your `dynex.ini` file and create `.env` instead:

```bash
rm dynex.ini

cat > .env << EOF
DYNEX_SDK_KEY=your_sdk_key
DYNEX_GRPC_ENDPOINT=127.0.0.1:9090
EOF
```

### Step 3: Update Code

**Before:**

```python
import dynex

model = dynex.BQM(bqm)
sampler = dynex.DynexSampler(model, )
sampleset = sampler.sample(num_reads=100, annealing_time=10, v2=True)
```

**After:**

```python
import dynex
from dynex import DynexConfig, ComputeBackend

config = DynexConfig(
    compute_backend=ComputeBackend.QPU,
    qpu_model=QPUModel.APOLLO_RC1  # Required for QPU (or use string "apollo_rc1")
)
model = dynex.BQM(bqm)
sampler = dynex.DynexSampler(model, config=config)
sampleset = sampler.sample(num_reads=100, annealing_time=500)
```

### Step 4: Verify Connectivity

- Ensure gRPC endpoint is accessible
- Test network connectivity to the backend
- Update firewall rules if needed for gRPC traffic (default port: 9090)

### Step 5: Test

```python
# Test basic functionality with your models
import dynex
from dynex import DynexConfig, ComputeBackend

# Use LOCAL for local testnet mode
config = DynexConfig(compute_backend=ComputeBackend.LOCAL)
# Or use CPU/GPU for network mode
# config = DynexConfig(compute_backend=ComputeBackend.CPU)

sampler = dynex.DynexSampler(model, config=config)
sampleset = sampler.sample(num_reads=10)
```

## Breaking Changes Summary

### Configuration

- `dynex.ini` no longer supported
- Environment variables required
- FTP configuration removed
- `mainnet` parameter removed (determined by backend)

### API

- `v2` parameter removed (v2 is default)
- `bnb` parameter removed
- `mainnet` parameter removed (backend determines mode)
- Backend selection now required via `ComputeBackend`
- **QPU model must be specified in `DynexConfig` (not in `sample()`)**
- `qpu_model` parameter removed from `sample()` method
- QPU model selection via `QPUModel` enum or string in config
- **`timeout` parameter moved to `DynexConfig.default_timeout` (not in sampler)**
- **`use_notebook_output` parameter moved to `DynexConfig` (not in sampler)**

### Communication

- HTTP REST API removed
- FTP protocol removed
- gRPC required

### Code Structure

- Monolithic `__init__.py` split into modules
- Import paths unchanged for backward compatibility

## New Configuration Parameters

SDK V2 introduces centralized configuration for environment and job parameters through `DynexConfig`.

### Environment Parameters (Config Only)

These parameters control the SDK environment and **must** be set via `DynexConfig`:

**1. `use_notebook_output` (default: `True`)**

Controls output display mode for Jupyter Notebook vs console:

```python
# Jupyter Notebook - dynamic table updates (default)
config = DynexConfig(
    compute_backend='cpu',
    use_notebook_output=True  # Tables update in-place
)

# Console/Script - static output
config = DynexConfig(
    compute_backend='cpu',
    use_notebook_output=False  # Each table printed separately
)
```

**Behavior:**
- `True`: Uses `clear_output()` for live table updates in Jupyter
- `False`: Prints each table update on new line (for scripts/console)

**2. `default_timeout` (default: `300.0`)**

Maximum wait time for solutions in seconds:

```python
config = DynexConfig(
    compute_backend='cpu',
    default_timeout=600.0  # Wait up to 10 minutes
)
sampler = DynexSampler(model, config=config)
# Timeout is set via config, cannot override in sampler
```

### Job Parameters (Config + Override)

These parameters have defaults in config but can be overridden per sampler:

**1. `default_description` (default: `"Dynex SDK Job"`)**

```python
# Set default in config
config = DynexConfig(
    compute_backend='cpu',
    default_description="My Project Jobs"
)

# Use default
sampler1 = DynexSampler(model, config=config)
# description = "My Project Jobs"

# Override for specific sampler
sampler2 = DynexSampler(model, config=config, description="Special Job")
# description = "Special Job"
```

**2. `preserve_solutions` (default: `False`)**

```python
# Set default in config
config = DynexConfig(
    compute_backend='cpu',
    preserve_solutions=True  # Keep all solution files
)

# Override for specific sampler
sampler = DynexSampler(model, config=config, preserve_solutions=False)
# This sampler will not preserve solutions
```

### Migration Example

**Legacy (v0.1.x):**
```python
sampler = dynex.DynexSampler(
    model,
    description="My Job",
    timeout=600,
    preserve_solutions=True,
    use_notebook_output=False
)
```

**SDK V2:**
```python
# Environment params in config (cannot override)
config = DynexConfig(
    compute_backend='cpu',
    use_notebook_output=False,  # For console
    default_timeout=600.0,       # 10 minutes
)

# Job params can override
sampler = DynexSampler(
    model,
    config=config,
    description="My Job",
    preserve_solutions=True
)
```

## Logging

SDK V2 provides comprehensive logging with platform identification, detailed timing, and progress tracking.

### Platform-Specific Logging

All log messages include platform identification in the prefix:

**Legacy:**
```
[DYNEX] INFO: SAMPLER INITIALISED
[DYNEX] INFO: STARTING JOB...
```

**SDK V2:**
```
INFO: [DYNEX-APOLLO-RC1] SAMPLER INITIALISED
INFO: [DYNEX-CPU] STARTING JOB...
INFO: [DYNEX-GPU] STARTING JOB...
```

The platform prefix (`APOLLO-RC1`, `CPU`, `GPU`, `LOCAL`) indicates where computation is executing.

### Enhanced Logging Features

**1. Problem Settings Summary:**

SDK V2 shows problem size and parameters at job start:

```python
model = dynex.BQM(bqm, logging=True)
sampler = dynex.DynexSampler(model, config=config)
sampleset = sampler.sample(num_reads=5, shots=3, annealing_time=100)
```

Output:
```
INFO: [DYNEX-APOLLO-RC1] Problem: 400 qubits, 15600 gates
INFO: [DYNEX-APOLLO-RC1] Settings: num_reads=5, shots=3, annealing_time=100
```

**2. Validation Warnings:**

Automatic warnings for potentially suboptimal parameters:

```
WARNING: [DYNEX-APOLLO-RC1] annealing_time=10 might be short for 400 qubits (recommended: >=100 for problems >100 qubits)
WARNING: [DYNEX-APOLLO-RC1] num_reads=100 is very high, consider reducing for faster testing
WARNING: [DYNEX-APOLLO-RC1] shots=50 is very high, consider reducing for faster testing
```

**3. Shot Progress Tracking:**

Real-time progress for multi-shot jobs:

**Legacy:**
```
[DYNEX] INFO: Processed 1 queued solutions
```

**SDK V2:**
```
INFO: [DYNEX-APOLLO-RC1] Shot 1/3 received
INFO: [DYNEX-APOLLO-RC1] Shot 2/3 received
INFO: [DYNEX-APOLLO-RC1] Shot 3/3 received
```

**4. Timing Breakdown:**

Detailed timing information at job completion:

```
INFO: [DYNEX-APOLLO-RC1] Average time per shot: 4.60s (3 shots in 13.80s)
INFO: [DYNEX-APOLLO-RC1] Timing breakdown:
INFO: [DYNEX-APOLLO-RC1]   Job upload:        0.34s
INFO: [DYNEX-APOLLO-RC1]   Time to 1st shot:  7.08s
INFO: [DYNEX-APOLLO-RC1]   Compute (Apollo):  13.80s
INFO: [DYNEX-APOLLO-RC1]   Solution download: 0.15s
INFO: [DYNEX-APOLLO-RC1]   Total elapsed:     14.29s
```

**5. Enhanced Result Table:**

**Legacy Table:**
```
╭────────────┬──────────┬─────────────────┬─────────────┬───────────┬────────────────┬────────────┬─────────┬────────────────╮
│   DYNEXJOB │   QUBITS │   QUANTUM GATES │   BLOCK FEE │   ELAPSED │   WORKERS READ │   CIRCUITS │   STEPS │   GROUND STATE │
├────────────┼──────────┼─────────────────┼─────────────┼───────────┼────────────────┼────────────┼─────────┼────────────────┤
│       3462 │      400 │           15600 │        0.00 │      0.06 │              1 │          1 │      -1 │         179.00 │
╰────────────┴──────────┴─────────────────┴─────────────┴───────────┴────────────────┴────────────┴─────────┴────────────────╯
```

**SDK V2 Table:**
```
╭────────────┬──────────┬─────────┬─────────────┬─────────┬────────────┬───────────┬───────────┬────────────────╮
│   DYNEXJOB │   QUBITS │   GATES │   NUM_READS │   SHOTS │   ANN.TIME │ ELAPSED   │   WORKERS │   GROUND STATE │
├────────────┼──────────┼─────────┼─────────────┼─────────┼────────────┼───────────┼───────────┼────────────────┤
│       3462 │      400 │   15600 │           5 │       3 │        100 │ 14.29s    │         3 │         179.00 │
╰────────────┴──────────┴─────────┴─────────────┴─────────┴────────────┴───────────┴───────────┴────────────────╯
```

**Changes:**
- Removed: `BLOCK FEE`, `CIRCUITS`, `STEPS` (deprecated columns)
- Added: `NUM_READS` (parallel samples per worker)
- Added: `SHOTS` (minimum solutions from network)
- Added: `ANN.TIME` (annealing time parameter)
- Renamed: `QUANTUM GATES` → `GATES`
- Renamed: `WORKERS READ` → `WORKERS`
- Format: `ELAPSED` now shows units (e.g., "14.29s")

### Log Levels

**INFO Level (Default):**
- Job lifecycle events
- Platform identification
- Settings summary
- Shot progress
- Timing breakdown
- Result tables

**DEBUG Level:**
- Internal state changes
- Solution processing details
- gRPC communication
- File operations

**WARNING Level:**
- Parameter validation warnings
- Scaling notifications
- Connectivity issues

**ERROR Level:**
- Job failures
- Configuration errors
- Communication errors

### Enabling Logging

```python
import logging

# INFO level - standard output
logging.basicConfig(level=logging.INFO)

# DEBUG level - detailed diagnostics
logging.basicConfig(level=logging.DEBUG)

# Enable logging in SDK
model = dynex.BQM(bqm, logging=True)
sampler = dynex.DynexSampler(model, config=config)
```

## Production Deployment Guide

### Infrastructure Requirements

**Minimum Requirements:**
- Python 3.11+
- 4 CPU cores
- 8GB RAM
- 10GB disk space
- Network: 100 Mbps with gRPC port access

**Recommended for Production:**
- Python 3.11+
- 8+ CPU cores
- 16GB+ RAM
- 50GB+ disk space (for solution caching)
- Network: 1 Gbps with stable gRPC endpoint connectivity

### Monitoring & Observability

**Metrics to Monitor:**

```python
import time
from prometheus_client import Counter, Histogram

# Define metrics
job_submissions = Counter('dynex_job_submissions_total', 'Total job submissions')
job_duration = Histogram('dynex_job_duration_seconds', 'Job execution time')
job_failures = Counter('dynex_job_failures_total', 'Total job failures')

# Instrument your code
def sample_with_monitoring(sampler, **kwargs):
    job_submissions.inc()
    start_time = time.time()
    
    try:
        result = sampler.sample(**kwargs)
        job_duration.observe(time.time() - start_time)
        return result
    except Exception as e:
        job_failures.inc()
        raise
```

**Log Aggregation:**

```python
import logging
from pythonjsonlogger import jsonlogger

# Configure JSON logging for centralized log aggregation
logger = logging.getLogger('dynex')
handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
handler.setFormatter(formatter)
logger.addHandler(handler)
```

### Container Deployment

**Dockerfile:**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Set environment variables
ENV DYNEX_SDK_KEY=${DYNEX_SDK_KEY}
ENV DYNEX_GRPC_ENDPOINT=${DYNEX_GRPC_ENDPOINT}

# Run application
CMD ["python", "your_app.py"]
```


## Testing Strategy

### Pre-Migration Testing

**1. Smoke Tests**

```python
def test_basic_functionality():
    """Quick verification that SDK V2 works"""
    config = DynexConfig(compute_backend=ComputeBackend.LOCAL)
    bqm = dimod.BinaryQuadraticModel({0: 1}, {(0, 1): -1}, 0.0, 'BINARY')
    model = dynex.BQM(bqm)
    sampler = dynex.DynexSampler(model, config=config)
    result = sampler.sample(num_reads=1)
    assert result is not None
    assert len(result) > 0
```

**2. Integration Tests**

```python
def test_end_to_end_workflow():
    """Test complete workflow with real backend"""
    config = DynexConfig(compute_backend=ComputeBackend.CPU)
    
    # Test BQM
    bqm = create_test_bqm()
    model = dynex.BQM(bqm)
    sampler = dynex.DynexSampler(model, config=config)
    result = sampler.sample(num_reads=10)
    
    assert result.first.energy < 0
    validate_solution(result.first.sample, bqm)
```

**3. Load Tests**

```python
import concurrent.futures

def load_test_concurrent_jobs(num_jobs=100):
    """Test system under concurrent load"""
    config = DynexConfig(compute_backend=ComputeBackend.CPU)
    
    def submit_job(job_id):
        sampler = dynex.DynexSampler(create_test_model(), config=config)
        return sampler.sample(num_reads=1)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(submit_job, i) for i in range(num_jobs)]
        results = [f.result() for f in futures]
    
    assert len(results) == num_jobs
    assert all(r is not None for r in results)
```

**4. Performance Regression Tests**

```python
import time

def test_performance_benchmarks():
    """Ensure V2 is not slower than legacy"""
    bqm = create_large_bqm(num_vars=1000)
    
    # Benchmark SDK V2
    start = time.time()
    config = DynexConfig(compute_backend=ComputeBackend.CPU)
    sampler = dynex.DynexSampler(dynex.BQM(bqm), config=config)
    result = sampler.sample(num_reads=10)
    v2_time = time.time() - start
    
    # Compare with baseline
    assert v2_time < BASELINE_TIME, f"Performance regression: {v2_time}s > {BASELINE_TIME}s"
```

### Post-Migration Validation

**Comparison Testing:**

```python
def validate_migration_results():
    """Compare legacy vs V2 results for identical inputs"""
    test_cases = load_test_cases()
    
    for test_case in test_cases:
        legacy_result = run_legacy_sdk(test_case)
        v2_result = run_sdk_v2(test_case)
        
        # Energies should be within tolerance
        assert abs(legacy_result.first.energy - v2_result.first.energy) < 0.01
        
        # Solution quality should be comparable
        assert are_solutions_equivalent(legacy_result, v2_result)
```

## Additional Resources

- SDK V2 Documentation: See README.md
- Example Code: See `examples/` directory
- API Reference: See source code docstrings
- Support Portal: support.dynex.com
- Community Forum: forum.dynex.com
- Enterprise Contact: enterprise@dynex.com

## Version Compatibility

| Feature | Legacy SDK | SDK V2 |
|---------|-----------|---------|
| BQM Support | Yes | Yes |
| CQM Support | Yes | Yes |
| DQM Support | Yes | Yes |
| SAT Support | Yes | No |
| HTTP API | Yes | No |
| FTP Transfer | Yes | No |
| gRPC Protocol | No | Yes |
| Environment Config | No | Yes |
| INI File Config | Yes | No |
| Compute Backend Selection | Implicit | Explicit |
| QPU Model Selection | No | Yes |
| v1/v2 Mode | Both | v2 Only |

This migration guide provides a comprehensive overview of changes between legacy SDK and SDK V2. For specific use cases not covered here, consult the source code or contact the development team.
