import time

import matplotlib.pyplot as plt
import numba
import numpy as np
from numba import njit, prange

# --- 1. Numba Optimized Sorting Implementations ---


@njit
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


@njit(parallel=True)
def bucket_sort(arr):
    n = len(arr)
    if n <= 1:
        return arr

    max_val, min_val = np.max(arr), np.min(arr)
    if max_val == min_val:
        return arr

    num_buckets = int(np.sqrt(n))
    rng = (max_val - min_val) / num_buckets

    # Pre-allocate buckets using a 2D array or list of arrays
    # Note: Numba's parallel support works best with fixed-shape arrays
    # For simplicity in this example, we use a list of arrays
    buckets = [np.zeros(0, dtype=arr.dtype) for _ in range(num_buckets + 1)]

    # Step 1: Distribute (Serial)
    for x in arr:
        index = int((x - min_val) / rng)
        # In a high-perf scenario, we'd pre-calculate bucket sizes
        # to avoid np.append reallocations
        buckets[index] = np.append(buckets[index], x)

    # Step 2: Parallel Sort (Meaningful performance gain here)
    for i in prange(len(buckets)):
        if len(buckets[i]) > 1:
            # We call a serial sort for the individual bucket
            buckets[i] = insertion_sort(buckets[i])

    # Step 3: Concatenate (Serial/Parallel hybrid)
    res = np.empty_like(arr)
    pos = 0
    for b in buckets:
        size = len(b)
        res[pos : pos + size] = b
        pos += size
    return res


@njit
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = arr[arr < pivot]
    middle = arr[arr == pivot]
    right = arr[arr > pivot]
    return np.concatenate((quicksort(left), middle, quicksort(right)))


@njit
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left, right = merge_sort(arr[:mid]), merge_sort(arr[mid:])
    res = np.empty_like(arr)
    i = j = k = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            res[k] = left[i]
            i += 1
        else:
            res[k] = right[j]
            j += 1
        k += 1
    res[k:] = left[i:] if i < len(left) else right[j:]
    return res


@njit
def radix_sort(arr):
    if len(arr) == 0:
        return arr
    mx, exp = np.max(arr), 1
    res = arr.copy()
    while mx // exp > 0:
        output = np.zeros_like(res)
        count = np.zeros(10, dtype=np.int64)
        for val in res:
            count[(val // exp) % 10] += 1
        for i in range(1, 10):
            count[i] += count[i - 1]
        for i in range(len(res) - 1, -1, -1):
            idx = (res[i] // exp) % 10
            output[count[idx] - 1] = res[i]
            count[idx] -= 1
        res = output.copy()
        exp *= 10
    return res


@njit(parallel=True)
def parallel_radix_sort(arr):
    if len(arr) == 0:
        return arr

    mx = np.max(arr)
    exp = 1
    res = arr.copy()
    n = len(res)
    num_threads = numba.config.NUMBA_NUM_THREADS  # Get available cores

    while mx // exp > 0:
        # 1. Parallel Local Counting
        # Each thread gets its own 10-slot count bucket
        local_counts = np.zeros((num_threads, 10), dtype=np.int64)
        chunk_size = (n + num_threads - 1) // num_threads

        for t in prange(num_threads):
            start = t * chunk_size
            end = min(start + chunk_size, n)
            for i in range(start, end):
                digit = (res[i] // exp) % 10
                local_counts[t, digit] += 1

        # 2. Global Offset Calculation (Serial, but very fast)
        # This calculates where each thread starts writing for each digit
        global_offsets = np.zeros((num_threads, 10), dtype=np.int64)
        total_digit_counts = np.zeros(10, dtype=np.int64)

        for d in range(10):
            for t in range(num_threads):
                global_offsets[t, d] = total_digit_counts[d]
                total_digit_counts[d] += local_counts[t, d]

        # Calculate final positions by prefix summing the digit totals
        digit_starts = np.zeros(10, dtype=np.int64)
        for d in range(1, 10):
            digit_starts[d] = digit_starts[d - 1] + total_digit_counts[d - 1]

        # 3. Parallel Scatter
        output = np.zeros_like(res)
        for t in prange(num_threads):
            start = t * chunk_size
            end = min(start + chunk_size, n)
            # Each thread knows its starting point for each digit
            thread_offsets = digit_starts.copy()
            for d in range(10):
                thread_offsets[d] += global_offsets[t, d]

            for i in range(start, end):
                digit = (res[i] // exp) % 10
                output[thread_offsets[digit]] = res[i]
                thread_offsets[digit] += 1

        res = output.copy()
        exp *= 10

    return res


@njit(parallel=True)
def parallel_timsort(arr):
    MIN_MERGE = 32
    n = len(arr)
    r, temp_n = 0, n
    while temp_n >= MIN_MERGE:
        r |= temp_n & 1
        temp_n >>= 1
    min_run = temp_n + r

    for start in range(0, n, min_run):
        end = min(start + min_run - 1, n - 1)
        for i in range(start + 1, end + 1):
            k, j = arr[i], i - 1
            while j >= start and arr[j] > k:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = k

    sz = min_run
    while sz < n:
        # The number of merges to perform at this size
        num_merges = (n + (2 * sz) - 1) // (2 * sz)

        # Parallelize the merging of independent chunks
        for m_idx in prange(num_merges):
            left = m_idx * 2 * sz
            mid = min(n - 1, left + sz - 1)
            right = min((left + 2 * sz - 1), (n - 1))

            if mid < right:
                # Merge logic (in-place or with temp array)
                l_arr = arr[left : mid + 1].copy()
                r_arr = arr[mid + 1 : right + 1].copy()
                i = j = 0
                k = left
                while i < len(l_arr) and j < len(r_arr):
                    if l_arr[i] <= r_arr[j]:
                        arr[k] = l_arr[i]
                        i += 1
                    else:
                        arr[k] = r_arr[j]
                        j += 1
                    k += 1
                while i < len(l_arr):
                    arr[k] = l_arr[i]
                    i += 1
                    k += 1
                while j < len(r_arr):
                    arr[k] = r_arr[j]
                    j += 1
                    k += 1
        sz *= 2
    return arr


@njit
def non_parallel_timsort(arr):
    MIN_MERGE = 32
    n = len(arr)
    r, temp_n = 0, n
    while temp_n >= MIN_MERGE:
        r |= temp_n & 1
        temp_n >>= 1
    min_run = temp_n + r

    for start in range(0, n, min_run):
        end = min(start + min_run - 1, n - 1)
        for i in range(start + 1, end + 1):
            k, j = arr[i], i - 1
            while j >= start and arr[j] > k:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = k

    sz = min_run
    while sz < n:
        # The number of merges to perform at this size
        num_merges = (n + (2 * sz) - 1) // (2 * sz)

        # Parallelize the merging of independent chunks
        for m_idx in range(num_merges):
            left = m_idx * 2 * sz
            mid = min(n - 1, left + sz - 1)
            right = min((left + 2 * sz - 1), (n - 1))

            if mid < right:
                # Merge logic (in-place or with temp array)
                l_arr = arr[left : mid + 1].copy()
                r_arr = arr[mid + 1 : right + 1].copy()
                i = j = 0
                k = left
                while i < len(l_arr) and j < len(r_arr):
                    if l_arr[i] <= r_arr[j]:
                        arr[k] = l_arr[i]
                        i += 1
                    else:
                        arr[k] = r_arr[j]
                        j += 1
                    k += 1
                while i < len(l_arr):
                    arr[k] = l_arr[i]
                    i += 1
                    k += 1
                while j < len(r_arr):
                    arr[k] = r_arr[j]
                    j += 1
                    k += 1
        sz *= 2
    return arr


@njit(parallel=True)
def parallel_radix_sort_optimized(arr):
    if len(arr) <= 1:
        return arr

    # 1. Base 256 for Bitwise Operations
    RADIX = 256
    MASK = 255

    mx = np.max(arr)

    # Determine number of 8-bit passes needed
    passes = 0
    temp_mx = mx
    while temp_mx > 0:
        passes += 1
        temp_mx >>= 8

    if passes == 0:
        passes = 1

    # 2. Pre-allocate "Ping-Pong" buffers to avoid in-loop allocations
    buffer1 = arr.copy()
    buffer2 = np.empty_like(arr)

    n = len(arr)
    num_threads = numba.config.NUMBA_NUM_THREADS

    for p in range(passes):
        shift = p * 8

        # --- Step 1: Parallel Local Counting ---
        local_counts = np.zeros((num_threads, RADIX), dtype=np.int64)
        chunk_size = (n + num_threads - 1) // num_threads

        for t in prange(num_threads):
            start = t * chunk_size
            end = min(start + chunk_size, n)
            for i in range(start, end):
                # Using fast bitwise logic instead of division/modulo
                digit = (buffer1[i] >> shift) & MASK
                local_counts[t, digit] += 1

        # --- Step 2: Global Offset Calculation (Serial) ---
        global_offsets = np.zeros((num_threads, RADIX), dtype=np.int64)
        total_digit_counts = np.zeros(RADIX, dtype=np.int64)

        for d in range(RADIX):
            for t in range(num_threads):
                global_offsets[t, d] = total_digit_counts[d]
                total_digit_counts[d] += local_counts[t, d]

        digit_starts = np.zeros(RADIX, dtype=np.int64)
        for d in range(1, RADIX):
            digit_starts[d] = digit_starts[d - 1] + total_digit_counts[d - 1]

        # --- Step 3: Parallel Scatter ---
        for t in prange(num_threads):
            start = t * chunk_size
            end = min(start + chunk_size, n)

            # Create local thread offsets cleanly
            thread_offsets = np.zeros(RADIX, dtype=np.int64)
            for d in range(RADIX):
                thread_offsets[d] = digit_starts[d] + global_offsets[t, d]

            for i in range(start, end):
                digit = (buffer1[i] >> shift) & MASK
                buffer2[thread_offsets[digit]] = buffer1[i]
                thread_offsets[digit] += 1

        # Swap buffers for the next pass
        temp = buffer1
        buffer1 = buffer2
        buffer2 = temp

    # After the loop, buffer1 always holds the final sorted array
    # because of the swap at the end of the last iteration.
    return buffer1


# --- 2. Benchmarking Engine ---


def run_benchmarks():
    start_overall = time.perf_counter()

    # You can push these higher now that it's JIT compiled
    sizes = list(
        range(1_000_000, 10_000_001, 1_000_000)
    )  # From 1K to 1M in steps of 10K
    # sizes = [100_000_000]  # Only benchmark the largest size for final results
    algs = [
        ("NumPy", lambda x: np.sort(x)),
        (
            "NumPy stable",
            lambda x: np.sort(x, kind="stable"),
        ),  # Using stable sort for better performance
        ("Timsort but better", parallel_timsort),
        ("Timsort(Non-Parallel)", non_parallel_timsort),
        ("Quicksort", quicksort),
        ("Merge Sort", merge_sort),
        ("Radix Sort", radix_sort),
        ("Radix Sort but better", parallel_radix_sort),
        ("Radix Sort but better but better", parallel_radix_sort_optimized),
        (
            "Default Python Sort",
            lambda x: x.sort(),
        ),  # In-place sort using Python's built-in Timsort
        # ("Bucket Sort", bucket_sort)
    ]

    # Warm-up (Compiles functions so compilation time isn't measured)
    dummy = np.random.randint(0, 100, 10)
    for _, func in algs:
        func(dummy.copy())

    results = {name: [] for name, _ in algs}

    print(f"Running benchmarks for size: {sizes[-1]}")

    for size in sizes:
        raw_data = np.random.randint(0, 10000, size=size)
        for name, func in algs:
            data_to_sort = raw_data.copy()
            start = time.perf_counter()
            func(data_to_sort)
            elapsed = time.perf_counter() - start
            results[name].append(elapsed)

    # --- Print Only Final Results ---
    largest_size = sizes[-1]
    print(f"\n{'=' * 50}")
    print(f"RESULTS FOR SIZE: {largest_size}")
    print(f"{'=' * 50}")
    print(f"{'Algorithm':<20} | {'Execution Time':<15}")
    print("-" * 38)

    end_overall = time.perf_counter()

    for name in results:
        print(f"{name:<20} | {results[name][-1]:.5f}s")

    print(f"Total Benchmark Time: {end_overall - start_overall:.2f} seconds\n")

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    for name, times in results.items():
        plt.plot(sizes, times, marker="o", label=name)

    plt.title(f"Sorting Performance (Up to {largest_size} elements)")
    plt.xlabel("Array Size")
    plt.ylabel("Seconds (Log Scale)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.show()


if __name__ == "__main__":
    run_benchmarks()
