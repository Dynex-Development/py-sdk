"""
Unit tests for dynex.utils module
"""

from dynex.utils import calculate_sha3_256_hash, calculate_sha3_256_hash_bin, max_value


def test_calculate_sha3_256_hash():
    """Test SHA3-256 hash calculation for strings"""
    test_string = "hello world"
    hash_result = calculate_sha3_256_hash(test_string)

    # SHA3-256 should produce 64 character hex string
    assert len(hash_result) == 64
    assert isinstance(hash_result, str)

    # Same input should produce same hash
    hash_result2 = calculate_sha3_256_hash(test_string)
    assert hash_result == hash_result2

    # Different input should produce different hash
    hash_result3 = calculate_sha3_256_hash("different")
    assert hash_result != hash_result3


def test_calculate_sha3_256_hash_bin():
    """Test SHA3-256 hash calculation for binary data"""
    test_bytes = b"hello world"
    hash_result = calculate_sha3_256_hash_bin(test_bytes)

    # SHA3-256 should produce 64 character hex string
    assert len(hash_result) == 64
    assert isinstance(hash_result, str)

    # Same input should produce same hash
    hash_result2 = calculate_sha3_256_hash_bin(test_bytes)
    assert hash_result == hash_result2

    # Different input should produce different hash
    hash_result3 = calculate_sha3_256_hash_bin(b"different")
    assert hash_result != hash_result3


def test_max_value():
    """Test max_value function"""
    input_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    result = max_value(input_list)
    assert result == 9

    input_list2 = [[10, 20], [5, 30], [15, 25]]
    result2 = max_value(input_list2)
    assert result2 == 30
