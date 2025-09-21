import psutil
import os
import time
import threading
from functools import wraps

def monitor_cpu_cores(func):
    """Decorator to monitor CPU core usage of a function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        
        # Get initial CPU info
        initial_cpu_percent = process.cpu_percent()
        cpu_count = psutil.cpu_count()
        
        print(f"Available CPU cores: {cpu_count}")
        print(f"Initial CPU usage: {initial_cpu_percent}%")
        
        # Monitor CPU usage during execution
        cpu_usage_data = []
        monitoring = True
        
        def monitor_cpu():
            while monitoring:
                cpu_percent = process.cpu_percent()
                cpu_usage_data.append(cpu_percent)
                time.sleep(0.1)  # Check every 100ms
        
        # Start monitoring
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Run the function
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Stop monitoring
        monitoring = False
        monitor_thread.join()
        
        # Calculate statistics
        avg_cpu = sum(cpu_usage_data) / len(cpu_usage_data) if cpu_usage_data else 0
        max_cpu = max(cpu_usage_data) if cpu_usage_data else 0
        
        print(f"Average CPU usage: {avg_cpu:.2f}%")
        print(f"Peak CPU usage: {max_cpu:.2f}%")
        print(f"Execution time: {end_time - start_time:.2f} seconds")
        
        return result
    
    return wrapper

# Example usage
@monitor_cpu_cores
def my_function():
    # Your function here
    data = [i**2 for i in range(1000000)]
    return sum(data)

result = my_function()

