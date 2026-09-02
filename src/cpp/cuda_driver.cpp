#include "cuda_driver.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace qupy::detail {
namespace {

using CUresult = int;
using CUdevice = int;
using CUdeviceptr = std::uint64_t;
using CUcontext = void*;
using CUmodule = void*;
using CUfunction = void*;
using CUstream = void*;
constexpr CUresult kCudaSuccess = 0;
[[maybe_unused]] constexpr std::string_view kSanitizerCudaDisabled =
    "CUDA driver execution is disabled in sanitizer builds";
using CuInit = CUresult (*)(unsigned int);
using CuDriverGetVersion = CUresult (*)(int*);
using CuDeviceGet = CUresult (*)(CUdevice*, int);
using CuDeviceGetCount = CUresult (*)(int*);
using CuDeviceGetName = CUresult (*)(char*, int, CUdevice);
using CuDeviceTotalMem = CUresult (*)(std::size_t*, CUdevice);
using CuDevicePrimaryCtxRetain = CUresult (*)(CUcontext*, CUdevice);
using CuDevicePrimaryCtxRelease = CUresult (*)(CUdevice);
using CuCtxSetCurrent = CUresult (*)(CUcontext);
using CuCtxSynchronize = CUresult (*)();
using CuModuleLoadDataEx = CUresult (*)(CUmodule*, const void*, unsigned int, int*, void**);
using CuModuleGetFunction = CUresult (*)(CUfunction*, CUmodule, const char*);
using CuModuleUnload = CUresult (*)(CUmodule);
using CuMemAlloc = CUresult (*)(CUdeviceptr*, std::size_t);
using CuMemFree = CUresult (*)(CUdeviceptr);
using CuMemsetD8 = CUresult (*)(CUdeviceptr, unsigned char, std::size_t);
using CuMemcpyHtoD = CUresult (*)(CUdeviceptr, const void*, std::size_t);
using CuMemcpyDtoH = CUresult (*)(void*, CUdeviceptr, std::size_t);
using CuLaunchKernel = CUresult (*)(
    CUfunction,
    unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int,
    unsigned int, CUstream, void**, void**
);
using CuGetErrorName = CUresult (*)(CUresult, const char**);
using CuGetErrorString = CUresult (*)(CUresult, const char**);

class DynamicLibrary {
public:
    DynamicLibrary();
    ~DynamicLibrary();
    DynamicLibrary(const DynamicLibrary&) = delete;
    DynamicLibrary& operator=(const DynamicLibrary&) = delete;

    [[nodiscard]] std::uintptr_t symbol(const char* name) const noexcept;

private:
#if defined(_WIN32)
    HMODULE handle_ = nullptr;
#else
    void* handle_ = nullptr;
#endif
};

DynamicLibrary::DynamicLibrary() {
#if defined(_WIN32)
    handle_ = LoadLibraryA("nvcuda.dll");
#else
    handle_ = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
    if (handle_ == nullptr) {
        throw std::runtime_error("CUDA driver library is not available");
    }
}

DynamicLibrary::~DynamicLibrary() {
#if defined(_WIN32)
    if (handle_ != nullptr) FreeLibrary(handle_);
#else
    if (handle_ != nullptr) dlclose(handle_);
#endif
}
std::uintptr_t DynamicLibrary::symbol(const char* name) const noexcept {
#if defined(_WIN32)
    return reinterpret_cast<std::uintptr_t>(GetProcAddress(handle_, name));
#else
    return reinterpret_cast<std::uintptr_t>(dlsym(handle_, name));
#endif
}

template <typename Function>
[[nodiscard]] Function load_symbol(
    const DynamicLibrary& library,
    const char* primary,
    const char* fallback = nullptr
) {
    std::uintptr_t raw = library.symbol(primary);
    if (raw == 0U && fallback != nullptr) raw = library.symbol(fallback);
    if (raw == 0U) {
        throw std::runtime_error(std::string("CUDA driver is missing symbol ") + primary);
    }
    Function result{};
    static_assert(sizeof(result) == sizeof(raw));
    std::memcpy(&result, &raw, sizeof(result));
    return result;
}

constexpr std::string_view kCudaPtx = R"ptx(
.version 7.0
.target sm_50
.address_size 64
.visible .entry apply_gate(
    .param .u64 p_state,
    .param .u64 p_dimension,
    .param .u32 p_kind,
    .param .u32 p_first,
    .param .u32 p_second,
    .param .f64 p_m00r,
    .param .f64 p_m00i,
    .param .f64 p_m01r,
    .param .f64 p_m01i,
    .param .f64 p_m10r,
    .param .f64 p_m10i,
    .param .f64 p_m11r,
    .param .f64 p_m11i
) {
    .reg .pred %p<10>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<24>;
    .reg .f64 %fd<40>;
    ld.param.u64 %rd1, [p_state];
    ld.param.u64 %rd2, [p_dimension];
    ld.param.u32 %r4, [p_kind];
    ld.param.u32 %r5, [p_first];
    ld.param.u32 %r6, [p_second];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mul.wide.u32 %rd3, %r1, %r2;
    cvt.u64.u32 %rd4, %r3;
    add.u64 %rd3, %rd3, %rd4;
    setp.eq.u32 %p1, %r4, 0;
    @%p1 bra SINGLE_GATE;
    setp.eq.u32 %p1, %r4, 1;
    @%p1 bra CX_GATE;
    setp.eq.u32 %p1, %r4, 2;
    @%p1 bra CZ_GATE;
    bra SWAP_GATE;

SINGLE_GATE:
    shr.u64 %rd4, %rd2, 1;
    setp.ge.u64 %p2, %rd3, %rd4;
    @%p2 bra DONE;
    mov.u64 %rd5, 1;
    shl.b64 %rd5, %rd5, %r5;
    sub.u64 %rd6, %rd5, 1;
    and.b64 %rd7, %rd3, %rd6;
    not.b64 %rd8, %rd6;
    and.b64 %rd9, %rd3, %rd8;
    shl.b64 %rd9, %rd9, 1;
    or.b64 %rd10, %rd7, %rd9;
    or.b64 %rd11, %rd10, %rd5;
    shl.b64 %rd10, %rd10, 4;
    shl.b64 %rd11, %rd11, 4;
    add.s64 %rd10, %rd1, %rd10;
    add.s64 %rd11, %rd1, %rd11;
    ld.global.f64 %fd0, [%rd10];
    ld.global.f64 %fd1, [%rd10+8];
    ld.global.f64 %fd2, [%rd11];
    ld.global.f64 %fd3, [%rd11+8];
    ld.param.f64 %fd4, [p_m00r];
    ld.param.f64 %fd5, [p_m00i];
    ld.param.f64 %fd6, [p_m01r];
    ld.param.f64 %fd7, [p_m01i];
    ld.param.f64 %fd8, [p_m10r];
    ld.param.f64 %fd9, [p_m10i];
    ld.param.f64 %fd10, [p_m11r];
    ld.param.f64 %fd11, [p_m11i];
    mul.rn.f64 %fd20, %fd4, %fd0;
    mul.rn.f64 %fd21, %fd5, %fd1;
    sub.rn.f64 %fd20, %fd20, %fd21;
    mul.rn.f64 %fd21, %fd6, %fd2;
    add.rn.f64 %fd20, %fd20, %fd21;
    mul.rn.f64 %fd21, %fd7, %fd3;
    sub.rn.f64 %fd20, %fd20, %fd21;
    mul.rn.f64 %fd22, %fd4, %fd1;
    mul.rn.f64 %fd23, %fd5, %fd0;
    add.rn.f64 %fd22, %fd22, %fd23;
    mul.rn.f64 %fd23, %fd6, %fd3;
    add.rn.f64 %fd22, %fd22, %fd23;
    mul.rn.f64 %fd23, %fd7, %fd2;
    add.rn.f64 %fd22, %fd22, %fd23;
    mul.rn.f64 %fd24, %fd8, %fd0;
    mul.rn.f64 %fd25, %fd9, %fd1;
    sub.rn.f64 %fd24, %fd24, %fd25;
    mul.rn.f64 %fd25, %fd10, %fd2;
    add.rn.f64 %fd24, %fd24, %fd25;
    mul.rn.f64 %fd25, %fd11, %fd3;
    sub.rn.f64 %fd24, %fd24, %fd25;
    mul.rn.f64 %fd26, %fd8, %fd1;
    mul.rn.f64 %fd27, %fd9, %fd0;
    add.rn.f64 %fd26, %fd26, %fd27;
    mul.rn.f64 %fd27, %fd10, %fd3;
    add.rn.f64 %fd26, %fd26, %fd27;
    mul.rn.f64 %fd27, %fd11, %fd2;
    add.rn.f64 %fd26, %fd26, %fd27;
    st.global.f64 [%rd10], %fd20;
    st.global.f64 [%rd10+8], %fd22;
    st.global.f64 [%rd11], %fd24;
    st.global.f64 [%rd11+8], %fd26;
    bra DONE;

CX_GATE:
    setp.ge.u64 %p2, %rd3, %rd2;
    @%p2 bra DONE;
    mov.u64 %rd5, 1;
    shl.b64 %rd6, %rd5, %r5;
    shl.b64 %rd7, %rd5, %r6;
    and.b64 %rd8, %rd3, %rd6;
    setp.eq.u64 %p3, %rd8, 0;
    @%p3 bra DONE;
    and.b64 %rd8, %rd3, %rd7;
    setp.ne.u64 %p4, %rd8, 0;
    @%p4 bra DONE;
    or.b64 %rd9, %rd3, %rd7;
    shl.b64 %rd10, %rd3, 4;
    shl.b64 %rd11, %rd9, 4;
    add.s64 %rd10, %rd1, %rd10;
    add.s64 %rd11, %rd1, %rd11;
    ld.global.f64 %fd0, [%rd10];
    ld.global.f64 %fd1, [%rd10+8];
    ld.global.f64 %fd2, [%rd11];
    ld.global.f64 %fd3, [%rd11+8];
    st.global.f64 [%rd10], %fd2;
    st.global.f64 [%rd10+8], %fd3;
    st.global.f64 [%rd11], %fd0;
    st.global.f64 [%rd11+8], %fd1;
    bra DONE;

CZ_GATE:
    setp.ge.u64 %p2, %rd3, %rd2;
    @%p2 bra DONE;
    mov.u64 %rd5, 1;
    shl.b64 %rd6, %rd5, %r5;
    shl.b64 %rd7, %rd5, %r6;
    and.b64 %rd8, %rd3, %rd6;
    setp.eq.u64 %p3, %rd8, 0;
    @%p3 bra DONE;
    and.b64 %rd8, %rd3, %rd7;
    setp.eq.u64 %p4, %rd8, 0;
    @%p4 bra DONE;
    shl.b64 %rd10, %rd3, 4;
    add.s64 %rd10, %rd1, %rd10;
    ld.global.f64 %fd0, [%rd10];
    ld.global.f64 %fd1, [%rd10+8];
    neg.f64 %fd0, %fd0;
    neg.f64 %fd1, %fd1;
    st.global.f64 [%rd10], %fd0;
    st.global.f64 [%rd10+8], %fd1;
    bra DONE;
SWAP_GATE:
    setp.ge.u64 %p2, %rd3, %rd2;
    @%p2 bra DONE;
    mov.u64 %rd5, 1;
    shl.b64 %rd6, %rd5, %r5;
    shl.b64 %rd7, %rd5, %r6;
    and.b64 %rd8, %rd3, %rd6;
    setp.ne.u64 %p3, %rd8, 0;
    @%p3 bra DONE;
    and.b64 %rd8, %rd3, %rd7;
    setp.eq.u64 %p4, %rd8, 0;
    @%p4 bra DONE;
    or.b64 %rd9, %rd6, %rd7;
    xor.b64 %rd9, %rd3, %rd9;
    shl.b64 %rd10, %rd3, 4;
    shl.b64 %rd11, %rd9, 4;
    add.s64 %rd10, %rd1, %rd10;
    add.s64 %rd11, %rd1, %rd11;
    ld.global.f64 %fd0, [%rd10];
    ld.global.f64 %fd1, [%rd10+8];
    ld.global.f64 %fd2, [%rd11];
    ld.global.f64 %fd3, [%rd11+8];
    st.global.f64 [%rd10], %fd2;
    st.global.f64 [%rd10+8], %fd3;
    st.global.f64 [%rd11], %fd0;
    st.global.f64 [%rd11+8], %fd1;
DONE:
    ret;
}
)ptx";

[[nodiscard]] std::uint32_t checked_u32(std::size_t value, const char* label) {
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error(std::string(label) + " exceeds CUDA kernel range");
    }
    return static_cast<std::uint32_t>(value);
}

[[nodiscard]] std::size_t checked_state_bytes(std::size_t num_qubits) {
    if (num_qubits >= std::numeric_limits<std::size_t>::digits) {
        throw std::length_error("CUDA state vector exceeds native address space");
    }
    const std::size_t dimension = std::size_t{1} << num_qubits;
    if (dimension > std::numeric_limits<std::size_t>::max() / sizeof(Complex)) {
        throw std::length_error("CUDA state vector exceeds native address space");
    }
    return dimension * sizeof(Complex);
}

class CudaRuntime {
public:
    CudaRuntime();
    ~CudaRuntime();
    CudaRuntime(const CudaRuntime&) = delete;
    CudaRuntime& operator=(const CudaRuntime&) = delete;

    [[nodiscard]] const std::string& device_name() const noexcept { return device_name_; }
    [[nodiscard]] int driver_version() const noexcept { return driver_version_; }
    [[nodiscard]] std::size_t total_memory() const noexcept { return total_memory_; }
    [[nodiscard, maybe_unused]] std::vector<Complex> statevector(
        std::size_t num_qubits,
        const std::vector<CudaStep>& steps
    );

private:
    void check(CUresult result, std::string_view operation) const;
    void set_current() const;
    void launch(CUdeviceptr state, std::uint64_t dimension, const CudaStep& step) const;

    DynamicLibrary library_;
    CuInit cu_init_;
    CuDriverGetVersion cu_driver_get_version_;
    CuDeviceGet cu_device_get_;
    CuDeviceGetCount cu_device_get_count_;
    CuDeviceGetName cu_device_get_name_;
    CuDeviceTotalMem cu_device_total_mem_;
    CuDevicePrimaryCtxRetain cu_primary_retain_;
    CuDevicePrimaryCtxRelease cu_primary_release_;
    CuCtxSetCurrent cu_ctx_set_current_;
    CuCtxSynchronize cu_ctx_synchronize_;
    CuModuleLoadDataEx cu_module_load_;
    CuModuleGetFunction cu_module_get_function_;
    CuModuleUnload cu_module_unload_;
    CuMemAlloc cu_mem_alloc_;
    CuMemFree cu_mem_free_;
    CuMemsetD8 cu_memset_d8_;
    CuMemcpyHtoD cu_memcpy_htod_;
    CuMemcpyDtoH cu_memcpy_dtoh_;
    CuLaunchKernel cu_launch_kernel_;
    CuGetErrorName cu_get_error_name_;
    CuGetErrorString cu_get_error_string_;
    CUdevice device_ = 0;
    CUcontext context_ = nullptr;
    CUmodule module_ = nullptr;
    CUfunction apply_gate_ = nullptr;
    std::string device_name_;
    int driver_version_ = 0;
    std::size_t total_memory_ = 0U;
    std::mutex execution_mutex_;
    CUdeviceptr workspace_ = 0U;
    std::size_t workspace_bytes_ = 0U;
};

CudaRuntime::CudaRuntime()
    : cu_init_(load_symbol<CuInit>(library_, "cuInit")),
      cu_driver_get_version_(load_symbol<CuDriverGetVersion>(library_, "cuDriverGetVersion")),
      cu_device_get_(load_symbol<CuDeviceGet>(library_, "cuDeviceGet")),
      cu_device_get_count_(load_symbol<CuDeviceGetCount>(library_, "cuDeviceGetCount")),
      cu_device_get_name_(load_symbol<CuDeviceGetName>(library_, "cuDeviceGetName")),
      cu_device_total_mem_(load_symbol<CuDeviceTotalMem>(library_, "cuDeviceTotalMem_v2", "cuDeviceTotalMem")),
      cu_primary_retain_(load_symbol<CuDevicePrimaryCtxRetain>(library_, "cuDevicePrimaryCtxRetain")),
      cu_primary_release_(load_symbol<CuDevicePrimaryCtxRelease>(library_, "cuDevicePrimaryCtxRelease_v2", "cuDevicePrimaryCtxRelease")),
      cu_ctx_set_current_(load_symbol<CuCtxSetCurrent>(library_, "cuCtxSetCurrent")),
      cu_ctx_synchronize_(load_symbol<CuCtxSynchronize>(library_, "cuCtxSynchronize")),
      cu_module_load_(load_symbol<CuModuleLoadDataEx>(library_, "cuModuleLoadDataEx")),
      cu_module_get_function_(load_symbol<CuModuleGetFunction>(library_, "cuModuleGetFunction")),
      cu_module_unload_(load_symbol<CuModuleUnload>(library_, "cuModuleUnload")),
      cu_mem_alloc_(load_symbol<CuMemAlloc>(library_, "cuMemAlloc_v2", "cuMemAlloc")),
      cu_mem_free_(load_symbol<CuMemFree>(library_, "cuMemFree_v2", "cuMemFree")),
      cu_memset_d8_(load_symbol<CuMemsetD8>(library_, "cuMemsetD8_v2", "cuMemsetD8")),
      cu_memcpy_htod_(load_symbol<CuMemcpyHtoD>(library_, "cuMemcpyHtoD_v2", "cuMemcpyHtoD")),
      cu_memcpy_dtoh_(load_symbol<CuMemcpyDtoH>(library_, "cuMemcpyDtoH_v2", "cuMemcpyDtoH")),
      cu_launch_kernel_(load_symbol<CuLaunchKernel>(library_, "cuLaunchKernel")),
      cu_get_error_name_(load_symbol<CuGetErrorName>(library_, "cuGetErrorName")),
      cu_get_error_string_(load_symbol<CuGetErrorString>(library_, "cuGetErrorString")) {
    check(cu_init_(0U), "cuInit");
    check(cu_driver_get_version_(&driver_version_), "cuDriverGetVersion");
    int device_count = 0;
    check(cu_device_get_count_(&device_count), "cuDeviceGetCount");
    if (device_count < 1) throw std::runtime_error("CUDA driver reports no devices");
    check(cu_device_get_(&device_, 0), "cuDeviceGet");
    std::array<char, 256> name{};
    check(cu_device_get_name_(name.data(), static_cast<int>(name.size()), device_), "cuDeviceGetName");
    device_name_ = name.data();
    check(cu_device_total_mem_(&total_memory_, device_), "cuDeviceTotalMem");
    check(cu_primary_retain_(&context_, device_), "cuDevicePrimaryCtxRetain");
    try {
        set_current();
        check(cu_module_load_(&module_, kCudaPtx.data(), 0U, nullptr, nullptr), "cuModuleLoadDataEx");
        check(cu_module_get_function_(&apply_gate_, module_, "apply_gate"), "cuModuleGetFunction");
    } catch (...) {
        (void)cu_primary_release_(device_);
        context_ = nullptr;
        throw;
    }
}

CudaRuntime::~CudaRuntime() {
    if (context_ != nullptr) {
        (void)cu_ctx_set_current_(context_);
        if (workspace_ != 0U) (void)cu_mem_free_(workspace_);
        if (module_ != nullptr) (void)cu_module_unload_(module_);
        (void)cu_primary_release_(device_);
    }
}

void CudaRuntime::check(CUresult result, std::string_view operation) const {
    if (result == kCudaSuccess) return;
    const char* name = nullptr;
    const char* description = nullptr;
    (void)cu_get_error_name_(result, &name);
    (void)cu_get_error_string_(result, &description);
    std::string message(operation);
    message += " failed";
    if (name != nullptr) message += std::string(": ") + name;
    if (description != nullptr) message += std::string(" (") + description + ')';
    throw std::runtime_error(message);
}

void CudaRuntime::set_current() const {
    check(cu_ctx_set_current_(context_), "cuCtxSetCurrent");
}
void CudaRuntime::launch(
    CUdeviceptr state,
    std::uint64_t dimension,
    const CudaStep& step
) const {
    constexpr std::uint32_t block_size = 256U;
    const std::uint64_t work_items = step.kind == CudaStepKind::Single
        ? dimension / 2U : dimension;
    const std::uint64_t grid64 = (work_items + block_size - 1U) / block_size;
    if (grid64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("CUDA launch grid exceeds driver range");
    }
    std::uint32_t kind = static_cast<std::uint32_t>(step.kind);
    std::uint32_t first = checked_u32(step.first, "CUDA qubit index");
    std::uint32_t second = checked_u32(step.second, "CUDA qubit index");
    double m00r = step.matrix[0].real();
    double m00i = step.matrix[0].imag();
    double m01r = step.matrix[1].real();
    double m01i = step.matrix[1].imag();
    double m10r = step.matrix[2].real();
    double m10i = step.matrix[2].imag();
    double m11r = step.matrix[3].real();
    double m11i = step.matrix[3].imag();
    void* arguments[] = {
        &state, &dimension, &kind, &first, &second,
        &m00r, &m00i, &m01r, &m01i,
        &m10r, &m10i, &m11r, &m11i,
    };
    check(
        cu_launch_kernel_(
            apply_gate_, static_cast<std::uint32_t>(grid64), 1U, 1U,
            block_size, 1U, 1U, 0U, nullptr, arguments, nullptr
        ),
        "cuLaunchKernel"
    );
}

[[maybe_unused]] std::vector<Complex> CudaRuntime::statevector(
    std::size_t num_qubits,
    const std::vector<CudaStep>& steps
) {
    static_assert(sizeof(Complex) == 2U * sizeof(double));
    const std::size_t bytes = checked_state_bytes(num_qubits);
    if (bytes > total_memory_) {
        throw std::length_error("CUDA state vector exceeds device memory");
    }
    const std::uint64_t dimension = static_cast<std::uint64_t>(bytes / sizeof(Complex));
    std::scoped_lock lock(execution_mutex_);
    set_current();
    if (workspace_bytes_ < bytes) {
        if (workspace_ != 0U) {
            check(cu_mem_free_(workspace_), "cuMemFree");
            workspace_ = 0U;
            workspace_bytes_ = 0U;
        }
        check(cu_mem_alloc_(&workspace_, bytes), "cuMemAlloc");
        workspace_bytes_ = bytes;
    }
    check(cu_memset_d8_(workspace_, 0U, bytes), "cuMemsetD8");
    const Complex zero_state{1.0, 0.0};
    check(cu_memcpy_htod_(workspace_, &zero_state, sizeof(zero_state)), "cuMemcpyHtoD");
    for (const CudaStep& step : steps) launch(workspace_, dimension, step);
    check(cu_ctx_synchronize_(), "cuCtxSynchronize");
    std::vector<Complex> result(static_cast<std::size_t>(dimension));
    check(cu_memcpy_dtoh_(result.data(), workspace_, bytes), "cuMemcpyDtoH");
    return result;
}

struct RuntimeHolder {
    RuntimeHolder() noexcept {
        try {
            runtime = std::make_unique<CudaRuntime>();
        } catch (const std::exception& error) {
            reason = error.what();
        } catch (...) {
            reason = "CUDA initialization failed with an unknown error";
        }
    }
    std::unique_ptr<CudaRuntime> runtime;
    std::string reason;
};

RuntimeHolder& holder() {
    static RuntimeHolder instance;
    return instance;
}

[[maybe_unused]] CudaRuntime& runtime() {
    RuntimeHolder& state = holder();
    if (state.runtime == nullptr) {
        throw std::runtime_error(
            state.reason.empty() ? "CUDA runtime is not available" : state.reason
        );
    }
    return *state.runtime;
}

}  // namespace

bool cuda_available() noexcept {
#ifdef QUPY_SANITIZER_BUILD
    return false;
#else
    return holder().runtime != nullptr;
#endif
}

std::string cuda_unavailable_reason() {
#ifdef QUPY_SANITIZER_BUILD
    return std::string(kSanitizerCudaDisabled);
#else
    return holder().reason;
#endif
}

std::string cuda_device_name() {
#ifdef QUPY_SANITIZER_BUILD
    throw std::runtime_error(std::string(kSanitizerCudaDisabled));
#else
    return runtime().device_name();
#endif
}

int cuda_driver_version() {
#ifdef QUPY_SANITIZER_BUILD
    throw std::runtime_error(std::string(kSanitizerCudaDisabled));
#else
    return runtime().driver_version();
#endif
}

std::size_t cuda_total_memory_bytes() {
#ifdef QUPY_SANITIZER_BUILD
    throw std::runtime_error(std::string(kSanitizerCudaDisabled));
#else
    return runtime().total_memory();
#endif
}

std::vector<Complex> cuda_statevector(
    std::size_t num_qubits,
    const std::vector<CudaStep>& steps
) {
#ifdef QUPY_SANITIZER_BUILD
    static_cast<void>(num_qubits);
    static_cast<void>(steps);
    throw std::runtime_error(std::string(kSanitizerCudaDisabled));
#else
    return runtime().statevector(num_qubits, steps);
#endif
}

}  // namespace qupy::detail
