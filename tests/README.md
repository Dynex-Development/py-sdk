# Tests

## Running tests

```bash
make test-unit          # unit tests (no credentials required)
make test-integration   # integration tests (requires .env with credentials)
make test               # all tests
```

Or directly:

```bash
uv run pytest tests/unit/ -v
uv run pytest tests/integration/ -v
```

## Test types

- **Unit tests** (`tests/unit/`): 150 tests, no API calls, always run in CI
- **Integration tests** (`tests/integration/`): 41 tests, require live API credentials

## Credentials for integration tests

Copy `.env.example` to `.env` and fill in your credentials:

```bash
DYNEX_SDK_KEY=your_sdk_key
DYNEX_GRPC_ENDPOINT=grpc.dynex.com:9090
```
