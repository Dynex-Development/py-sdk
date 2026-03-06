"""Unit tests for _DynexSampler helper methods."""

from dynex.sampler import _DynexSampler


def test_check_list_length_empty():
    """Test _check_list_length with empty list."""
    assert _DynexSampler._check_list_length([]) is False


def test_check_list_length_short_sublists():
    """Test _check_list_length with sublists <= 3."""
    assert _DynexSampler._check_list_length([[1], [1, 2], [1, 2, 3]]) is False


def test_check_list_length_long_sublist():
    """Test _check_list_length with sublist > 3."""
    assert _DynexSampler._check_list_length([[1, 2, 3, 4]]) is True
    assert _DynexSampler._check_list_length([[1], [1, 2, 3, 4, 5]]) is True


def test_sanitize_solution_name_empty():
    """Test _sanitize_solution_name with empty string."""
    result = _DynexSampler._sanitize_solution_name("")
    assert result == "solution"


def test_sanitize_solution_name_none():
    """Test _sanitize_solution_name with None."""
    result = _DynexSampler._sanitize_solution_name(None)
    assert result == "solution"


def test_sanitize_solution_name_valid():
    """Test _sanitize_solution_name with valid filename."""
    result = _DynexSampler._sanitize_solution_name("my_solution.txt")
    assert result == "my_solution.txt"


def test_sanitize_solution_name_invalid_chars():
    """Test _sanitize_solution_name with invalid characters."""
    result = _DynexSampler._sanitize_solution_name("file@#$%name")
    assert result == "file____name"


def test_parse_solution_subject_empty():
    """Test _parse_solution_subject with empty string."""
    result = _DynexSampler._parse_solution_subject("")
    assert result == {}


def test_parse_solution_subject_none():
    """Test _parse_solution_subject with None."""
    result = _DynexSampler._parse_solution_subject(None)
    assert result == {}


def test_parse_solution_subject_valid():
    """Test _parse_solution_subject with valid subject string."""
    subject = "job_id=123;worker_id=456;chip=apollo"
    result = _DynexSampler._parse_solution_subject(subject)
    assert result["job_id"] == "123"
    assert result["worker_id"] == "456"
    assert result["chip"] == "apollo"


def test_parse_solution_subject_partial():
    """Test _parse_solution_subject with partial data."""
    subject = "job_id=789"
    result = _DynexSampler._parse_solution_subject(subject)
    assert result["job_id"] == "789"
    assert len(result) == 1


def test_parse_solution_subject_json():
    """Test _parse_solution_subject with JSON string."""
    subject = '{"job_id": "123", "worker_id": "456"}'
    result = _DynexSampler._parse_solution_subject(subject)
    assert result["job_id"] == "123"
    assert result["worker_id"] == "456"


def test_parse_solution_numbers_empty():
    """Test _parse_solution_numbers with empty string."""
    result = _DynexSampler._parse_solution_numbers("")
    assert result == {}


def test_parse_solution_numbers_none():
    """Test _parse_solution_numbers with None."""
    result = _DynexSampler._parse_solution_numbers(None)
    assert result == {}


def test_parse_solution_numbers_valid():
    """Test _parse_solution_numbers with valid number string."""
    text = "1 100 50 -25.5"
    result = _DynexSampler._parse_solution_numbers(text)
    assert result["chips"] == 1
    assert result["steps"] == 100
    assert result["loc"] == 50
    assert result["energy"] == -25.5


def test_parse_solution_numbers_insufficient():
    """Test _parse_solution_numbers with insufficient numbers."""
    text = "1 100"  # Only 2 numbers, need at least 4
    result = _DynexSampler._parse_solution_numbers(text)
    assert result == {}


def test_coerce_int_none():
    """Test _coerce_int with None."""
    assert _DynexSampler._coerce_int(None) is None


def test_coerce_int_empty_string():
    """Test _coerce_int with empty string."""
    assert _DynexSampler._coerce_int("") is None


def test_coerce_int_valid_string():
    """Test _coerce_int with valid string."""
    assert _DynexSampler._coerce_int("42") == 42
    assert _DynexSampler._coerce_int("0") == 0
    assert _DynexSampler._coerce_int("-10") == -10


def test_coerce_int_valid_int():
    """Test _coerce_int with valid int."""
    assert _DynexSampler._coerce_int(42) == 42


def test_coerce_int_valid_float_string():
    """Test _coerce_int with float string."""
    assert _DynexSampler._coerce_int("42.9") == 42


def test_coerce_float_none():
    """Test _coerce_float with None."""
    assert _DynexSampler._coerce_float(None) is None


def test_coerce_float_empty_string():
    """Test _coerce_float with empty string."""
    assert _DynexSampler._coerce_float("") is None


def test_coerce_float_valid_string():
    """Test _coerce_float with valid string."""
    assert _DynexSampler._coerce_float("3.14") == 3.14
    assert _DynexSampler._coerce_float("0.0") == 0.0
    assert _DynexSampler._coerce_float("-2.5") == -2.5


def test_coerce_float_valid_float():
    """Test _coerce_float with valid float."""
    assert _DynexSampler._coerce_float(3.14) == 3.14


def test_decompress_bytes_none_compression():
    """Test _decompress_bytes with no compression."""
    data = b"test data"
    result = _DynexSampler._decompress_bytes(data, None)
    assert result == data


def test_decompress_bytes_empty_compression():
    """Test _decompress_bytes with empty compression string."""
    data = b"test data"
    result = _DynexSampler._decompress_bytes(data, "")
    assert result == data


def test_decode_varint_single_byte():
    """Test _decode_varint with single byte value."""
    buffer = b"\x08"  # varint 8
    value, next_index = _DynexSampler._decode_varint(buffer, 0)
    assert value == 8
    assert next_index == 1


def test_decode_varint_multi_byte():
    """Test _decode_varint with multi-byte value."""
    buffer = b"\xac\x02"  # varint 300
    value, next_index = _DynexSampler._decode_varint(buffer, 0)
    assert value == 300
    assert next_index == 2


def test_decode_varint_large_value():
    """Test _decode_varint with large value."""
    buffer = b"\xff\xff\xff\xff\x0f"  # varint 4294967295
    value, next_index = _DynexSampler._decode_varint(buffer, 0)
    assert value == 4294967295
    assert next_index == 5


def test_extract_voltage_values_empty():
    """Test _extract_voltage_values with empty line."""
    result = _DynexSampler._extract_voltage_values("")
    assert result == ["NaN"]


def test_extract_voltage_values_no_commas():
    """Test _extract_voltage_values with line without commas."""
    line = "1 0 1 0 1"  # No commas, should return NaN
    result = _DynexSampler._extract_voltage_values(line)
    assert result == ["NaN"]


def test_extract_voltage_values_with_commas():
    """Test _extract_voltage_values with comma-separated values."""
    line = "1,0,1,0,1"
    result = _DynexSampler._extract_voltage_values(line)
    assert result == ["1", "0", "1", "0", "1"]


def test_extract_voltage_values_multiline():
    """Test _extract_voltage_values with multiline data."""
    line = "energy,0.5,2.3\n1,0,1,0"
    result = _DynexSampler._extract_voltage_values(line)
    assert result == ["energy", "0.5", "2.3"]


def test_extract_voltage_values_prefer_last():
    """Test _extract_voltage_values with prefer_last."""
    line = "1,0,1\n0,1,0"
    result = _DynexSampler._extract_voltage_values(line, prefer_last=True)
    assert result == ["0", "1", "0"]


def test_ensure_voltage_text_already_str():
    """Test _ensure_voltage_text with string."""
    result = _DynexSampler._ensure_voltage_text("1,0,1")
    assert result == "1,0,1"


def test_ensure_voltage_text_bytes():
    """Test _ensure_voltage_text with bytes."""
    result = _DynexSampler._ensure_voltage_text(b"1,0,1")
    assert result == "1,0,1"


def test_ensure_voltage_text_empty():
    """Test _ensure_voltage_text with empty input."""
    result = _DynexSampler._ensure_voltage_text("")
    assert result == ""


def test_convert_list_to_dict():
    """Test _convert with list input (converts pairs to dict)."""
    input_list = [0, 1, 1, 0, 2, 1]
    result = _DynexSampler._convert(input_list)
    assert result == {0: 1, 1: 0, 2: 1}


def test_convert_tuple_to_dict():
    """Test _convert with tuple input."""
    input_tuple = (0, 1, 1, 0)
    result = _DynexSampler._convert(input_tuple)
    assert result == {0: 1, 1: 0}


def test_convert_empty():
    """Test _convert with empty input."""
    result = _DynexSampler._convert([])
    assert result == {}
