# write a warpper function to check the function cost time
import time


def timeit(func):
    """
    A decorator to measure the execution time of a function.

    Args:
        func: The function to be decorated.

    Returns:
        wrapper: The wrapped function that measures execution time.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result

    return wrapper
