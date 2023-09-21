import functools
import time

def logger(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("=============================" + func.__name__ + ": start=============================")
        output = func(*args, **kwargs)
        print("=============================" + func.__name__ + ": end=============================")
        return output
    return wrapper

def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} executed in {execution_time} seconds")
        return result
    return wrapper

def deprecated(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Function {func.__name__} is deprecated.")
        print("This function is deprecated and should not be used.")
        return func(*args, **kwargs)
    return wrapper


@deprecated
@timeit
@logger
def test_func():
    ans = 0
    for i in range(100):
        ans += i
    print(ans)

if __name__ == "__main__":
    test_func()