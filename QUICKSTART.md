# Dynex SDK — Quick Start

## Installation

```bash
pip install dynex
```

Or from source with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/tapok1999/dynex-sdk
cd dynex-sdk
uv sync
```

## Configure credentials

```bash
# .env
DYNEX_SDK_KEY=your_sdk_key_here
DYNEX_GRPC_ENDPOINT=quantum-router-engine-grpc.hz.dynex.co:3000
```

## Running examples

```bash
# N-Queens problem
python qpu_examples/nqueens_demo.py

# Traveling Salesman Problem
python qpu_examples/tsp_demo.py

# Integer Factorization
python qpu_examples/factorization_demo.py

# Protein Folding
python qpu_examples/protein_folding_demo.py

# Grover's Algorithm
python qpu_examples/grover_demo.py

# Shor's Algorithm
python qpu_examples/shor_demo.py

# Full Adder Circuit
python qpu_examples/adder_demo.py
```

## Jupyter notebooks

See [github.com/Dynex-Development/awesome-dynex](https://github.com/Dynex-Development/awesome-dynex) for a full collection of example notebooks.

## Documentation

- Full documentation: [docs.dynexcoin.org](https://docs.dynexcoin.org)
- Migration guide: `MIGRATION.md`
- API reference: source docstrings

## Support

For issues and questions: https://github.com/tapok1999/dynex-sdk/issues
