// Standalone necessary-condition filter; the caller must confirm primality and N.
// Build: nvcc -O3 -std=c++17 -arch=sm_86 -shared -Xcompiler -fPIC \
//        tools/psp_pairs_gpu.cu -o bin/psp_pairs_gpu.so
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>

extern "C" {
struct PspRange {
    uint64_t p, first, step, count, offset;
};
}
static_assert(sizeof(PspRange) == 40, "PspRange ABI requires five uint64_t words");

namespace {
constexpr uint64_t kSafeLimit = uint64_t{1} << 63;
constexpr uint32_t kMaxJobs = 1u << 20;
constexpr unsigned kBlockSize = 256;
thread_local char error_text[2048] = {};

struct Failure {};

struct Context {
    int device = 0;
    uint64_t capacity = 0;
    uint64_t max_blocks = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    PspRange* ranges = nullptr;
    uint32_t range_capacity = 0;
    uint64_t* indices = nullptr;
    unsigned long long* hits = nullptr;
};

void check(cudaError_t code, const char* operation) {
    if (code != cudaSuccess) {
        std::snprintf(error_text, sizeof(error_text), "%s: %s (%d)",
                      operation, cudaGetErrorString(code), static_cast<int>(code));
        throw Failure{};
    }
}

void require(bool valid, const char* message) {
    if (!valid) {
        std::snprintf(error_text, sizeof(error_text), "%s", message);
        throw Failure{};
    }
}

bool cleanup_check(cudaError_t code, const char* operation) noexcept {
    if (code == cudaSuccess) return true;
    const size_t used = std::strlen(error_text);
    std::snprintf(error_text + used, sizeof(error_text) - used, "%s%s: %s (%d)",
                  used ? "; " : "", operation, cudaGetErrorString(code),
                  static_cast<int>(code));
    return false;
}

// Cleanup continues after errors so one failed release cannot hide later leaks.
bool release(Context* ctx) noexcept {
    if (!ctx) return true;
    bool ok = cleanup_check(cudaSetDevice(ctx->device), "cleanup cudaSetDevice");
    if (ok) {
        if (ctx->stream)
            ok = cleanup_check(cudaStreamSynchronize(ctx->stream),
                               "cleanup cudaStreamSynchronize") && ok;
        if (ctx->ranges)
            ok = cleanup_check(cudaFree(ctx->ranges), "cleanup cudaFree ranges") && ok;
        if (ctx->indices)
            ok = cleanup_check(cudaFree(ctx->indices), "cleanup cudaFree indices") && ok;
        if (ctx->hits)
            ok = cleanup_check(cudaFree(ctx->hits), "cleanup cudaFree hits") && ok;
        if (ctx->start)
            ok = cleanup_check(cudaEventDestroy(ctx->start),
                               "cleanup cudaEventDestroy start") && ok;
        if (ctx->stop)
            ok = cleanup_check(cudaEventDestroy(ctx->stop),
                               "cleanup cudaEventDestroy stop") && ok;
        if (ctx->stream)
            ok = cleanup_check(cudaStreamDestroy(ctx->stream),
                               "cleanup cudaStreamDestroy") && ok;
    }
    delete ctx;
    return ok;
}

void drain(Context* ctx, bool queued) noexcept {
    if (queued && ctx && ctx->stream &&
        cleanup_check(cudaSetDevice(ctx->device), "error cudaSetDevice")) {
        cleanup_check(cudaStreamSynchronize(ctx->stream), "error cudaStreamSynchronize");
    }
}

__device__ __forceinline__ uint64_t mont_inverse(uint64_t q) {
    // The odd-modulus seed has five correct bits; four Newton steps reach 64.
    uint64_t x = (3 * q) ^ 2;
#pragma unroll
    for (int i = 0; i < 4; ++i) x *= 2 - q * x;
    return 0 - x;
}

__device__ __forceinline__ uint64_t mont_square(uint64_t x, uint64_t q,
                                               uint64_t inverse) {
    const uint64_t low = x * x;
    const uint64_t high = __umul64hi(x, x);
    const uint64_t m = low * inverse;
    const uint64_t cancelled = low + m * q;
    // q < 2^63 and x < q imply REDC < 2q < 2^64: no high-word overflow.
    const uint64_t reduced = high + __umul64hi(m, q) + (cancelled < low);
    return reduced >= q ? reduced - q : reduced;
}

__device__ __forceinline__ uint64_t mod_double(uint64_t x, uint64_t q) {
    const uint64_t doubled = x + x;  // x < q < 2^63, so this cannot overflow.
    return doubled >= q ? doubled - q : doubled;
}

__device__ __forceinline__ bool passes(uint64_t p, uint64_t q) {
    const uint64_t inverse = mont_inverse(q);
    // R mod q = ((R - 1) mod q + 1) mod q, with no representation of R.
    uint64_t one = UINT64_MAX % q + 1;
    if (one == q) one = 0;
    const uint64_t exponent = p - 1;
    uint64_t value = mod_double(one, q);
    for (int bit = 62 - __clzll(exponent); bit >= 0; --bit) {
        value = mont_square(value, q, inverse);
        if ((exponent >> bit) & 1) value = mod_double(value, q);
    }
    return value == one;
}

__global__ void filter_kernel(const PspRange* ranges, uint32_t job_count,
                              uint64_t total, uint64_t* indices,
                              unsigned long long* hits) {
    const uint64_t index = uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (index >= total) return;

    uint32_t lo = 0, hi = job_count;
    while (lo + 1 < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (ranges[mid].offset <= index) lo = mid;
        else hi = mid;
    }
    const PspRange job = ranges[lo];
    const uint64_t q = job.first + (index - job.offset) * job.step;
    if (passes(job.p, q)) {
        const unsigned long long slot = atomicAdd(hits, 1ULL);
        // Each thread emits at most once; total <= capacity, so none are dropped.
        indices[slot] = index;
    }
}

void validate(const Context* ctx, const PspRange* jobs, uint32_t job_count,
              uint64_t total, const uint64_t* host_indices,
              const uint64_t* hit_count, const float* kernel_ms) {
    require(ctx != nullptr, "psp_gpu_filter: null context");
    require(hit_count && kernel_ms, "psp_gpu_filter: hit_count and kernel_ms are required");
    require(total <= ctx->capacity, "psp_gpu_filter: total exceeds context capacity");
    require(job_count <= kMaxJobs, "psp_gpu_filter: job_count exceeds 1048576");
    require(job_count == 0 || jobs, "psp_gpu_filter: null jobs");
    require(total == 0 || host_indices, "psp_gpu_filter: null host_indices");
    require((total == 0) == (job_count == 0),
            "psp_gpu_filter: empty total and empty jobs must agree");
    require(total / kBlockSize + (total % kBlockSize != 0) <= ctx->max_blocks,
            "psp_gpu_filter: total exceeds device grid limit");
    uint64_t next_offset = 0;
    for (uint32_t i = 0; i < job_count; ++i) {
        const PspRange& job = jobs[i];
        const char* reason = nullptr;
        if (job.p < 3 || job.p >= kSafeLimit) reason = "p must be in [3, 2^63)";
        else if (!job.count) reason = "count must be nonzero";
        else if (job.offset != next_offset) reason = "offsets must be contiguous from zero";
        else if (job.count > total - next_offset) reason = "counts exceed total";
        else if (!job.step) reason = "step must be nonzero";
        else if (job.first < 3 || job.first >= kSafeLimit || !(job.first & 1))
            reason = "first must be odd and in [3, 2^63)";
        else if (job.count > 1 && (job.step & 1))
            reason = "step must be even unless count is one";
        else {
            const __uint128_t last = __uint128_t{job.first} +
                __uint128_t{job.count - 1} * job.step;
            if (last >= kSafeLimit) reason = "last q must be below 2^63";
        }
        if (reason) {
            std::snprintf(error_text, sizeof(error_text), "psp_gpu_filter: job %u: %s",
                          i, reason);
            throw Failure{};
        }
        next_offset += job.count;
    }
    require(next_offset == total, "psp_gpu_filter: sum of counts does not equal total");
}

void reserve_ranges(Context* ctx, uint32_t job_count) {
    if (job_count <= ctx->range_capacity) return;
    uint32_t capacity = ctx->range_capacity ? ctx->range_capacity : 1;
    while (capacity < job_count) capacity *= 2;
    // Only the small descriptor buffer grows; index and counter buffers persist.
    if (ctx->ranges) {
        check(cudaFree(ctx->ranges), "cudaFree old ranges");
        ctx->ranges = nullptr;
        ctx->range_capacity = 0;
    }
    check(cudaMalloc(reinterpret_cast<void**>(&ctx->ranges),
                     size_t{capacity} * sizeof(PspRange)), "cudaMalloc ranges");
    ctx->range_capacity = capacity;
}
}  // namespace

// Calls select the context's device. Serialize use/destruction of a given context.
// Success is 0, failure is -1 (create returns null). Errors are thread-local and
// valid until the next operation on that thread. destroy consumes the handle even
// on a CUDA cleanup failure; destroy(nullptr) succeeds.
extern "C" void* psp_gpu_create(int device, uint64_t capacity) noexcept {
    error_text[0] = '\0';
    Context* ctx = nullptr;
    try {
        require(device >= 0, "psp_gpu_create: device must be nonnegative");
        require(capacity > 0, "psp_gpu_create: capacity must be nonzero");
        require(capacity <= std::numeric_limits<size_t>::max() / sizeof(uint64_t),
                "psp_gpu_create: capacity byte size overflows size_t");
        int device_count = 0;
        check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        require(device < device_count,
                "psp_gpu_create: device index exceeds available device count");
        check(cudaSetDevice(device), "cudaSetDevice");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
        require(properties.maxThreadsPerBlock >= static_cast<int>(kBlockSize) &&
                properties.maxThreadsDim[0] >= static_cast<int>(kBlockSize) &&
                properties.maxGridSize[0] > 0,
                "psp_gpu_create: device does not support the required 256-thread grid");
        ctx = new Context;
        ctx->device = device;
        ctx->capacity = capacity;
        ctx->max_blocks = static_cast<uint64_t>(properties.maxGridSize[0]);
        check(cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking),
              "cudaStreamCreateWithFlags");
        check(cudaEventCreate(&ctx->start), "cudaEventCreate start");
        check(cudaEventCreate(&ctx->stop), "cudaEventCreate stop");
        check(cudaMalloc(reinterpret_cast<void**>(&ctx->indices),
                         size_t{capacity} * sizeof(uint64_t)), "cudaMalloc indices");
        check(cudaMalloc(reinterpret_cast<void**>(&ctx->hits),
                         sizeof(unsigned long long)), "cudaMalloc hits");
        return ctx;
    } catch (const Failure&) {
    } catch (const std::exception& e) {
        std::snprintf(error_text, sizeof(error_text), "psp_gpu_create: %s", e.what());
    } catch (...) {
        std::snprintf(error_text, sizeof(error_text), "psp_gpu_create: unknown exception");
    }
    release(ctx);
    return nullptr;
}

// host_indices must have room for total words. Output order is nondeterministic;
// indices are flattened offsets, not q values. Count/timing are zeroed on failure;
// ignore host_indices on failure because a failed transfer may partially write it.
// kernel_ms excludes transfers and setup. No GPU work is issued for an empty batch.
extern "C" int psp_gpu_filter(void* handle, const PspRange* jobs, uint32_t job_count,
                              uint64_t total, uint64_t* host_indices,
                              uint64_t* hit_count, float* kernel_ms) noexcept {
    error_text[0] = '\0';
    if (hit_count) *hit_count = 0;
    if (kernel_ms) *kernel_ms = 0;
    Context* ctx = static_cast<Context*>(handle);
    bool queued = false;
    // Keep the async copy destination alive through error-path stream draining.
    unsigned long long hits = 0;
    try {
        validate(ctx, jobs, job_count, total, host_indices, hit_count, kernel_ms);
        if (!total) return 0;
        check(cudaSetDevice(ctx->device), "cudaSetDevice");
        reserve_ranges(ctx, job_count);
        queued = true;
        check(cudaMemcpyAsync(ctx->ranges, jobs, size_t{job_count} * sizeof(PspRange),
                              cudaMemcpyHostToDevice, ctx->stream), "cudaMemcpyAsync ranges");
        check(cudaMemsetAsync(ctx->hits, 0, sizeof(unsigned long long), ctx->stream),
              "cudaMemsetAsync hits");
        check(cudaEventRecord(ctx->start, ctx->stream), "cudaEventRecord start");
        const unsigned blocks = static_cast<unsigned>(
            total / kBlockSize + (total % kBlockSize != 0));
        filter_kernel<<<blocks, kBlockSize, 0, ctx->stream>>>(
            ctx->ranges, job_count, total, ctx->indices, ctx->hits);
        check(cudaGetLastError(), "filter_kernel launch");
        check(cudaEventRecord(ctx->stop, ctx->stream), "cudaEventRecord stop");
        check(cudaMemcpyAsync(&hits, ctx->hits, sizeof(hits), cudaMemcpyDeviceToHost,
                              ctx->stream), "cudaMemcpyAsync hit count");
        check(cudaStreamSynchronize(ctx->stream), "cudaStreamSynchronize kernel");
        require(hits <= total, "psp_gpu_filter: GPU hit count exceeds total");
        float elapsed = 0;
        check(cudaEventElapsedTime(&elapsed, ctx->start, ctx->stop), "cudaEventElapsedTime");
        if (hits) {
            check(cudaMemcpyAsync(host_indices, ctx->indices,
                                  size_t{hits} * sizeof(uint64_t), cudaMemcpyDeviceToHost,
                                  ctx->stream), "cudaMemcpyAsync surviving indices");
            check(cudaStreamSynchronize(ctx->stream), "cudaStreamSynchronize indices");
        }
        *hit_count = static_cast<uint64_t>(hits);
        *kernel_ms = elapsed;
        return 0;
    } catch (const Failure&) {
    } catch (const std::exception& e) {
        std::snprintf(error_text, sizeof(error_text), "psp_gpu_filter: %s", e.what());
    } catch (...) {
        std::snprintf(error_text, sizeof(error_text), "psp_gpu_filter: unknown exception");
    }
    drain(ctx, queued);
    return -1;
}

extern "C" int psp_gpu_destroy(void* handle) noexcept {
    error_text[0] = '\0';
    try {
        return release(static_cast<Context*>(handle)) ? 0 : -1;
    } catch (const std::exception& e) {
        std::snprintf(error_text, sizeof(error_text), "psp_gpu_destroy: %s", e.what());
    } catch (...) {
        std::snprintf(error_text, sizeof(error_text), "psp_gpu_destroy: unknown exception");
    }
    return -1;
}

extern "C" const char* psp_gpu_error() noexcept {
    return error_text;
}
