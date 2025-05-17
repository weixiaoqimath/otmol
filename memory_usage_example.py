#!/usr/bin/env python
# -*- coding: utf-8 -*-

import resource
import time
import random
import platform
import sys
import gc
import logging
import pdb  # Python debugger
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_memory_usage():
    """Get memory usage in KB, handling platform differences"""
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On macOS, ru_maxrss is in bytes, on Linux it's in KB
    if platform.system() == 'Darwin':  # macOS
        return mem / 1024.0  # Convert bytes to KB
    return mem  # Already in KB on Linux

def print_memory_details(call_number, memory_used):
    """Print detailed memory information"""
    print(f"\nCall #{call_number}:")
    print(f"Memory used: {memory_used:.2f} KB")
    print(f"Memory used: {memory_used/1024:.2f} MB")

# Method 1: Using resource module (built-in)
def measure_memory_usage(func):
    def wrapper(*args, **kwargs):
        # Get memory usage before function call
        start_mem = get_memory_usage()
        
        # Call the function
        result = func(*args, **kwargs)
        
        # Get memory usage after function call
        end_mem = get_memory_usage()
        
        # Calculate memory difference (in KB)
        memory_used = end_mem - start_mem
        return result, memory_used
    return wrapper

# Example function to test memory usage
@measure_memory_usage
def main(size):
    # Simulate some work
    arr = np.ones(size)
    result = np.sum(arr)
    return result

# Example function to test memory usage with NumPy
@measure_memory_usage
def create_numpy_array(size):
    return np.ones(size)

# Example function to test memory usage with Python list
@measure_memory_usage
def create_python_list(size):
    return [1.0 for _ in range(size)]

def debug_example():
    # Example of using pdb programmatically
    print("Starting debug example")
    x = 10
    y = 20
    
    # Set a breakpoint
    pdb.set_trace()  # Execution will pause here
    
    z = x + y
    print(f"Result: {z}")

# Method 2: Using memory_profiler (requires installation)
# To use this method, you need to:
# 1. Install memory_profiler: pip install memory_profiler
# 2. Uncomment the following code and run with: python -m memory_profiler memory_usage_example.py

"""
from memory_profiler import profile

@profile
def create_large_list_with_profile(size):
    return [random.random() for _ in range(size)]
"""

if __name__ == "__main__":
    print("Platform:", platform.system())
    print("Python version:", sys.version)
    
    # Run main() multiple times
    size = 1000000
    for i in range(5):
        result, memory_used = main(size)
        print_memory_details(i + 1, memory_used)
        
        # Force garbage collection between runs
        gc.collect()
    
    print("\nTesting NumPy array memory usage:")
    numpy_array = create_numpy_array(size)
    
    print("\nTesting Python list memory usage:")
    python_list = create_python_list(size)
    
    # Clean up
    del numpy_array
    del python_list
    
    # Uncomment to test debug_example
    # debug_example()
    
    # Test with memory_profiler (uncomment to use)
    # print("\nTesting with memory_profiler:")
    # large_list = create_large_list_with_profile(1000000)
    
    # Demonstrate memory cleanup
    print("\nAfter deleting the lists:")
    gc.collect()
    print_memory_details("Memory after cleanup", get_memory_usage()) 