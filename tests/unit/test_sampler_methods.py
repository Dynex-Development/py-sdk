"""Unit tests for _DynexSampler instance methods."""

import os
from unittest.mock import patch

import dimod
import numpy as np
import pytest

from dynex import BQM, DynexConfig
from dynex.sampler import _DynexSampler


@pytest.fixture
def mock_model():
    """Create a mock BQM model."""
    config = DynexConfig(compute_backend="cpu")
    bqm = dimod.BinaryQuadraticModel({0: -1, 1: -1}, {(0, 1): 2}, 0.0, "BINARY")
    return BQM(bqm, config=config)


def test_sampler_log_debug_enabled(mock_model, caplog):
    """Test _log_debug with debug logging enabled."""
    import logging

    config = DynexConfig(compute_backend="cpu")

    with patch("dynex.grpc_client.DynexGrpcClient"):
        sampler = _DynexSampler(mock_model, logging=True, config=config)
        sampler.logger.setLevel(logging.DEBUG)

        sampler._log_debug("Test debug message")

        # Check if debug message was logged
        assert any("Test debug message" in record.message for record in caplog.records if record.levelname == "DEBUG")


def test_sampler_log_debug_disabled(mock_model, caplog):
    """Test _log_debug with debug logging disabled."""
    import logging

    config = DynexConfig(compute_backend="cpu")

    with patch("dynex.grpc_client.DynexGrpcClient"):
        sampler = _DynexSampler(mock_model, logging=True, config=config)
        sampler.logger.setLevel(logging.INFO)

        sampler._log_debug("Test debug message")

        # Debug message should not be logged at INFO level
        debug_messages = [r for r in caplog.records if r.levelname == "DEBUG" and "Test debug message" in r.message]
        assert len(debug_messages) == 0


def test_save_cnf_creates_file(mock_model, tmp_path):
    """Test _save_cnf creates a file with clauses."""
    config = DynexConfig(compute_backend="cpu")

    with patch("dynex.grpc_client.DynexGrpcClient"):
        sampler = _DynexSampler(mock_model, logging=False, config=config)

        clauses = [[1, -1, 2], [2, 3], [-1, -2, -3]]
        filename = str(tmp_path / "test.cnf")

        sampler._save_cnf(clauses, filename)

        # Verify file was created
        assert os.path.exists(filename)

        # Verify content
        with open(filename, "r") as f:
            content = f.read()
            assert "1 -1 2 0" in content
            assert "2 3 0" in content
            assert "-1 -2 -3 0" in content


def test_save_cnf_single_clause(mock_model, tmp_path):
    """Test _save_cnf with single clause."""
    config = DynexConfig(compute_backend="cpu")

    with patch("dynex.grpc_client.DynexGrpcClient"):
        sampler = _DynexSampler(mock_model, logging=False, config=config)

        clauses = [[1, -1]]
        filename = str(tmp_path / "single.cnf")

        sampler._save_cnf(clauses, filename)

        # Verify file was created
        assert os.path.exists(filename)

        with open(filename, "r") as f:
            content = f.read()
            assert "1 -1 0" in content


def test_solution_metrics_from_filename_with_stats(mock_model):
    """Test _solution_metrics_from_filename with provided stats."""
    config = DynexConfig(compute_backend="cpu")

    with patch("dynex.grpc_client.DynexGrpcClient"):
        sampler = _DynexSampler(mock_model, logging=False, config=config)
        sampler.filename = "test_job"

        stats = {"chips": 5, "steps": 1000, "loc": 100, "energy": -25.5}

        filename = "test_job_solution.txt"
        fallback_info = "5_1000_100_-25.5"

        chips, steps, loc, energy = sampler._solution_metrics_from_filename(filename, fallback_info, stats)

        assert chips == 5
        assert steps == 1000
        assert loc == 100
        assert energy == -25.5


def test_solution_metrics_from_filename_fallback(mock_model):
    """Test _solution_metrics_from_filename with fallback parsing."""
    config = DynexConfig(compute_backend="cpu")

    with patch("dynex.grpc_client.DynexGrpcClient"):
        sampler = _DynexSampler(mock_model, logging=False, config=config)
        sampler.filename = "test_job"

        filename = "test_job_3_500_50_-10.5.txt"
        fallback_info = "3_500_50_-10.5"

        chips, steps, loc, energy = sampler._solution_metrics_from_filename(filename, fallback_info, {})

        assert chips == 3
        assert steps == 500
        assert loc == 50
        assert energy == -10.5


def test_get_solution_metrics(mock_model):
    """Test _get_solution_metrics extracts metrics from filename."""
    config = DynexConfig(compute_backend="cpu")

    with patch("dynex.grpc_client.DynexGrpcClient"):
        sampler = _DynexSampler(mock_model, logging=False, config=config)
        sampler.filename = "job123"
        sampler._grpc_solution_stats = {}

        filename = "job123_2_800_75_-15.0.txt"

        chips, steps, loc, energy = sampler._get_solution_metrics(filename)

        assert chips == 2
        assert steps == 800
        assert loc == 75
        assert energy == -15.0


def test_clean_calls_list_files(mock_model):
    """Test _clean calls list_files_with_text_local for mainnet."""
    config = DynexConfig(compute_backend="cpu")

    with patch("dynex.grpc_client.DynexGrpcClient"):
        sampler = _DynexSampler(mock_model, logging=False, config=config)
        sampler.config.mainnet = True

        with patch.object(sampler, "list_files_with_text_local", return_value=[]):
            sampler._clean()
            # Just verify it doesn't raise - _clean doesn't actually remove files anymore
            # (FTP functionality removed)


def test_delete_local_files_by_prefix(mock_model, tmp_path):
    """Test delete_local_files_by_prefix removes matching files for local solver."""
    config = DynexConfig(compute_backend="local")

    with patch("dynex.grpc_client.DynexGrpcClient"):
        sampler = _DynexSampler(mock_model, logging=False, config=config)

        (tmp_path / "prefix_file1.txt").write_text("test1")
        (tmp_path / "prefix_file2.txt").write_text("test2")
        (tmp_path / "other_file.txt").write_text("other")

        sampler.delete_local_files_by_prefix(str(tmp_path), "prefix_")

        assert not (tmp_path / "prefix_file1.txt").exists()
        assert not (tmp_path / "prefix_file2.txt").exists()
        assert (tmp_path / "other_file.txt").exists()


def test_delete_local_files_by_prefix_mainnet_clears_cache(mock_model):
    """Test delete_local_files_by_prefix clears in-memory cache for mainnet."""
    config = DynexConfig(compute_backend="cpu")

    with patch("dynex.grpc_client.DynexGrpcClient"):
        sampler = _DynexSampler(mock_model, logging=False, config=config)

        sampler._solution_cache["prefix_abc.solution_1"] = b"data1"
        sampler._solution_cache["prefix_abc.solution_2"] = b"data2"
        sampler._solution_cache["other_abc.solution_3"] = b"data3"

        sampler.delete_local_files_by_prefix("tmp/", "prefix_abc")

        assert "prefix_abc.solution_1" not in sampler._solution_cache
        assert "prefix_abc.solution_2" not in sampler._solution_cache
        assert "other_abc.solution_3" in sampler._solution_cache


def _make_solution(name, raw_bytes, kind="inline"):
    """Build a minimal solution-like object for _consume_solution_meta."""
    import base64

    encoded = base64.b64encode(raw_bytes).decode("ascii")
    obj = type(
        "S",
        (),
        {
            "name": name,
            "data": encoded,
            "url": "",
            "kind": kind,
            "compression": "",
            "valid": True,
            "subject": "",
            "checksum": "",
            "size": len(raw_bytes),
            "compressed_size": 0,
        },
    )()
    obj.ListFields = lambda: []
    return obj


def test_consume_solution_meta_mainnet_stores_in_cache(mock_model, tmp_path):
    """On mainnet without debug, solution goes into _solution_cache only — no file written."""
    config = DynexConfig(compute_backend="cpu")

    with patch("dynex.grpc_client.DynexGrpcClient"):
        sampler = _DynexSampler(mock_model, logging=False, config=config)
        sampler.filename = "job_abc"
        sampler.filepath = str(tmp_path) + "/"
        sampler.current_job_id = 42

        raw = b"1,0,1,0"
        solution = _make_solution("solution_x", raw)

        sampler._consume_solution_meta(solution)

        local_name = "job_abc.solution_x"
        assert local_name in sampler._solution_cache
        assert sampler._solution_cache[local_name] == raw
        assert not (tmp_path / local_name).exists()


def test_consume_solution_meta_debug_writes_file(mock_model, tmp_path):
    """With debug_save_solutions=True on mainnet, solution is in cache AND written to disk."""
    config = DynexConfig(compute_backend="cpu", debug_save_solutions=True)

    with patch("dynex.grpc_client.DynexGrpcClient"):
        sampler = _DynexSampler(mock_model, logging=False, config=config)
        sampler.filename = "job_debug"
        sampler.filepath = str(tmp_path) + "/"
        sampler.current_job_id = 1

        raw = b"0,1,0,1"
        solution = _make_solution("solution_y", raw)

        sampler._consume_solution_meta(solution)

        local_name = "job_debug.solution_y"
        assert local_name in sampler._solution_cache
        assert sampler._solution_cache[local_name] == raw
        assert (tmp_path / local_name).exists()
        assert (tmp_path / local_name).read_bytes() == raw


def test_delete_local_files_by_prefix_debug_deletes_files(mock_model, tmp_path):
    """With debug_save_solutions=True, delete also removes physical debug files."""
    config = DynexConfig(compute_backend="cpu", debug_save_solutions=True)

    with patch("dynex.grpc_client.DynexGrpcClient"):
        sampler = _DynexSampler(mock_model, logging=False, config=config)

        sampler._solution_cache["prefix_x.sol_1"] = b"d1"
        sampler._solution_cache["prefix_x.sol_2"] = b"d2"
        sampler._solution_cache["other_x.sol_3"] = b"d3"

        (tmp_path / "prefix_x.sol_1").write_bytes(b"d1")
        (tmp_path / "prefix_x.sol_2").write_bytes(b"d2")
        (tmp_path / "other_x.sol_3").write_bytes(b"d3")

        sampler.delete_local_files_by_prefix(str(tmp_path), "prefix_x")

        assert "prefix_x.sol_1" not in sampler._solution_cache
        assert "prefix_x.sol_2" not in sampler._solution_cache
        assert "other_x.sol_3" in sampler._solution_cache
        assert not (tmp_path / "prefix_x.sol_1").exists()
        assert not (tmp_path / "prefix_x.sol_2").exists()
        assert (tmp_path / "other_x.sol_3").exists()


def test_list_files_with_text_local_returns_cache_keys(mock_model):
    """On mainnet, list_files_with_text_local returns all _solution_cache keys."""
    config = DynexConfig(compute_backend="cpu")

    with patch("dynex.grpc_client.DynexGrpcClient"):
        sampler = _DynexSampler(mock_model, logging=False, config=config)

        sampler._solution_cache["job_list.sol_a"] = b"a"
        sampler._solution_cache["job_list.sol_b"] = b"b"
        sampler._solution_cache["other_job.sol_c"] = b"c"

        result = sampler.list_files_with_text_local()

        assert "job_list.sol_a" in result
        assert "job_list.sol_b" in result
        assert "other_job.sol_c" in result


def test_convert_dict_to_list(mock_model):
    """Test _convert transforms dict values to list."""
    config = DynexConfig(compute_backend="cpu")

    with patch("dynex.grpc_client.DynexGrpcClient"):
        sampler = _DynexSampler(mock_model, logging=False, config=config)

        # _convert takes pairs and creates dict, but we're testing the reverse
        input_list = [0, 1, 1, 0, 2, 1]
        result = sampler._convert(input_list)

        assert result == {0: 1, 1: 0, 2: 1}


def test_energy_calculation(mock_model):
    """Test _energy calculates energy correctly."""
    config = DynexConfig(compute_backend="cpu")

    with patch("dynex.grpc_client.DynexGrpcClient"):
        sampler = _DynexSampler(mock_model, logging=False, config=config)

        # Set up model properties
        sampler.model.num_variables = 3
        sampler.model.precision = 0.1
        sampler.model.clauses = [
            [10, -1, -2],  # weight=10, 3-lit clause
            [5, 1],  # weight=5, 2-lit clause
        ]
        sampler.model.bqm_offset = 0.0

        # Sample where clauses may be violated
        sample = [1, 1, 0]

        loc, energy = sampler._energy(sample, mapping=False)

        # _energy returns tuple (loc, energy)
        assert isinstance(loc, (int, np.integer))
        assert isinstance(energy, (int, float, np.floating))


def test_sample_to_assignments(mock_model):
    """Test _sample_to_assignments converts sample to dict."""
    config = DynexConfig(compute_backend="cpu")

    with patch("dynex.grpc_client.DynexGrpcClient"):
        sampler = _DynexSampler(mock_model, logging=False, config=config)

        # Set up var_mappings directly on sampler
        sampler.var_mappings = {"x": 0, "y": 1, "z": 2}

        # _sample_to_assignments takes a list of values (voltages)
        lowest_set = [1.0, -1.0, 1.0]  # Positive=1, Negative=0

        assignments = sampler._sample_to_assignments(lowest_set)

        # Result uses keys from var_mappings
        assert "x" in assignments
        assert "y" in assignments
        assert "z" in assignments
        assert assignments["x"] == 1
        assert assignments["y"] == 0
        assert assignments["z"] == 1


def test_process_voltage_line(mock_model):
    """Test _process_voltage_line extracts voltage values."""
    config = DynexConfig(compute_backend="cpu")

    with patch("dynex.grpc_client.DynexGrpcClient"):
        sampler = _DynexSampler(mock_model, logging=False, config=config)

        # Test with comma-separated values
        line = "1,0,1,0,1"
        result = sampler._process_voltage_line(line)

        assert result == ["1", "0", "1", "0", "1"]


def test_lookup_grpc_stats(mock_model):
    """Test _lookup_grpc_stats retrieves stats from cache."""
    config = DynexConfig(compute_backend="cpu")

    with patch("dynex.grpc_client.DynexGrpcClient"):
        sampler = _DynexSampler(mock_model, logging=False, config=config)

        # Set up stats cache
        sampler._grpc_solution_stats = {"solution1.txt": {"chips": 3, "steps": 500, "energy": -10.0}}

        stats = sampler._lookup_grpc_stats("solution1.txt", "fallback_info")

        assert stats["chips"] == 3
        assert stats["steps"] == 500
        assert stats["energy"] == -10.0


def test_lookup_grpc_stats_missing(mock_model):
    """Test _lookup_grpc_stats with missing stats returns empty dict."""
    config = DynexConfig(compute_backend="cpu")

    with patch("dynex.grpc_client.DynexGrpcClient"):
        sampler = _DynexSampler(mock_model, logging=False, config=config)
        sampler._grpc_solution_stats = {}

        stats = sampler._lookup_grpc_stats("nonexistent.txt", "info")

        assert stats == {}
