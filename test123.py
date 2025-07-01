# Performance-optimized version with memory and runtime improvements
import sys

# Use bytes for better memory efficiency if this represents binary/encoded data
DATA_BYTES = b'fhjfhgfhgwfdhAKHlfgv'

# String version with interning for frequent access optimization
# Use sys.intern() for string interning to save memory on repeated access
DATA_STRING = sys.intern('fhjfhgfhgwfdhAKHlfgv')

# Lazy loading function for even better performance when data isn't always needed
_cached_data = None

def get_data():
    """Get the data with lazy loading and caching for optimal performance."""
    global _cached_data
    if _cached_data is None:
        _cached_data = DATA_STRING
    return _cached_data

# Expose the optimized data as module constants to prevent recreation
__all__ = ['DATA_BYTES', 'DATA_STRING', 'get_data']