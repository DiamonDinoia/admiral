// snmalloc vs glibc per-call alloc+free across the scratch_alloc band: 128 KiB..32 MiB
// doubling, serial and 16 threads, median of 31 trials of K pairs per cell. The arms are
// the two paths scratch_alloc.cpp routes: snmalloc::libc::memalign/free and over-aligned
// ::operator new[]/delete[]. Re-derive kSnmallocSmallMaxBytes (or delete the glibc band)
// from this probe, never by nudging the constant until one cell passes.
//
// Build (repo root, GCC 14, snmalloc from the CPM cache the engine builds against):
//   g++ -std=c++20 -O2 -DNDEBUG -pthread -DSNMALLOC_HAS_LINUX_FUTEX_H \
//     -DSNMALLOC_HAS_LINUX_RANDOM_H -DSNMALLOC_NO_REALLOCARR -DSNMALLOC_NO_REALLOCARRAY \
//     -DSNMALLOC_PLATFORM_HAS_GETENTROPY -DSNMALLOC_PTHREAD_ATFORK_WORKS \
//     -DSNMALLOC_USE_WAIT_ON_ADDRESS=1 \
//     -isystem <cpm>/snmalloc/<rev>/src tools/alloc_band_probe.cpp -o alloc_band_probe
// Run (record load average and every other pinned job with the output):
//   taskset -c 8 ./alloc_band_probe 1   (serial arm)
//   taskset -c 8-23 ./alloc_band_probe 16 (threaded arm)
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <thread>
#include <vector>

#include <snmalloc/snmalloc.h>

namespace {

constexpr std::size_t kAlign = 64;

long long pair_ns(std::size_t bytes, bool use_sn, std::size_t iters) {
    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iters; ++i) {
        if (use_sn) {
            void* p = snmalloc::libc::memalign(kAlign, bytes);
            snmalloc::libc::free(p);
        } else {
            void* p = ::operator new[](bytes, std::align_val_t{kAlign});
            ::operator delete[](p, std::align_val_t{kAlign});
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() /
           static_cast<long long>(iters);
}

long long cell_median(std::size_t bytes, bool use_sn, std::size_t threads,
                      std::size_t iters) {
    std::vector<long long> trials;
    trials.reserve(31);
    if (threads == 1) {
        for (int t = 0; t < 31; ++t) trials.push_back(pair_ns(bytes, use_sn, iters));
    } else {
        for (int t = 0; t < 31; ++t) {
            std::atomic<std::size_t> ready{0};
            std::atomic<bool> go{false};
            std::vector<std::thread> pool;
            std::vector<long long> per(threads, 0);
            for (std::size_t w = 0; w < threads; ++w) {
                pool.emplace_back([&, w] {
                    ready.fetch_add(1, std::memory_order_relaxed);
                    while (!go.load(std::memory_order_acquire)) {}
                    per[w] = pair_ns(bytes, use_sn, iters);
                });
            }
            while (ready.load(std::memory_order_relaxed) != threads) {}
            const auto t0 = std::chrono::steady_clock::now();
            go.store(true, std::memory_order_release);
            for (auto& th : pool) th.join();
            const auto t1 = std::chrono::steady_clock::now();
            (void)t0;
            (void)t1;
            trials.push_back(*std::max_element(per.begin(), per.end()));
        }
    }
    std::sort(trials.begin(), trials.end());
    return trials[trials.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t threads = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 1;
    std::printf("# alloc_band_probe threads=%zu align=%zu trials=31 median of per-pair ns\n",
                threads, kAlign);
    std::printf("size_bytes,snmalloc_ns,glibc_ns\n");
    for (std::size_t bytes = 128u << 10; bytes <= (32u << 20); bytes *= 2) {
        const std::size_t iters = std::max<std::size_t>(4, (std::size_t{1} << 28) / bytes);
        for (int arm = 0; arm < 2; ++arm) (void)pair_ns(bytes, arm == 0, 8);  // warm
        const long long sn = cell_median(bytes, true, threads, iters);
        const long long gl = cell_median(bytes, false, threads, iters);
        std::printf("%zu,%lld,%lld\n", bytes, sn, gl);
        std::fflush(stdout);
    }
    return 0;
}
