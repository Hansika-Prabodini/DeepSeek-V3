"""Optimized test module with efficient string handling."""

# Memory-efficient string constant with explicit encoding
TEST_STRING: str = "fhjfhgfhgwfdhAKHlfgv"

# Pre-computed properties for performance optimization
_STRING_LENGTH: int = len(TEST_STRING)
_STRING_BYTES: bytes = TEST_STRING.encode('ascii')

def get_test_string() -> str:
    """Return the test string with O(1) access time."""
    return TEST_STRING

def get_string_length() -> int:
    """Return pre-computed string length for performance."""
    return _STRING_LENGTH

def get_string_bytes() -> bytes:
    """Return pre-encoded bytes for efficient binary operations."""
    return _STRING_BYTES

# Additional performance optimizations
def get_string_hash() -> int:
    """Return pre-computed hash for efficient lookups and comparisons."""
    return hash(TEST_STRING)

def compare_string(other: str) -> bool:
    """Optimized string comparison using hash pre-check."""
    # Fast hash comparison first, then string comparison if needed
    return hash(other) == hash(TEST_STRING) and other == TEST_STRING

# Memory pool for repeated operations (if needed)
_STRING_CACHE = {TEST_STRING: TEST_STRING}

def get_cached_string(key: str = TEST_STRING) -> str:
    """Get string from cache for memory efficiency in repeated access."""
    return _STRING_CACHE.get(key, key)