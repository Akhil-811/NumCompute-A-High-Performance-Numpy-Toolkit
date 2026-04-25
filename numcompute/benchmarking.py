import time


def benchmark(func, *args, repeat=5):
    times = []
    for _ in range(repeat):
        start = time.time()
        func(*args)
        times.append(time.time() - start)
    return min(times)