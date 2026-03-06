from dynex import ComputeBackend


def test_compute_backend_values():
    """Test ComputeBackend enum values."""
    assert ComputeBackend.UNSPECIFIED.value == "unspecified"
    assert ComputeBackend.CPU.value == "cpu"
    assert ComputeBackend.GPU.value == "gpu"
    assert ComputeBackend.LOCAL.value == "local"
    assert ComputeBackend.QPU.value == "qpu"


def test_compute_backend_string_conversion():
    """Test ComputeBackend can be used as string."""
    backend = ComputeBackend.QPU
    assert backend.value == "qpu"
    assert backend == "qpu"


def test_compute_backend_comparison():
    """Test ComputeBackend comparison."""
    assert ComputeBackend.CPU == "cpu"
    assert ComputeBackend.GPU != ComputeBackend.CPU
    assert ComputeBackend.QPU == ComputeBackend.QPU


def test_compute_backend_iteration():
    """Test iterating over ComputeBackend values."""
    backends = [b.value for b in ComputeBackend]
    assert "unspecified" in backends
    assert "cpu" in backends
    assert "gpu" in backends
    assert "local" in backends
    assert "qpu" in backends
    assert len(backends) == 5
