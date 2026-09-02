#include "qupy/advanced.hpp"
#include "qupy/provider_abi.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace qupy {
namespace {

constexpr std::size_t kInitialJobIdCapacity = 256U;
constexpr std::size_t kInitialErrorCapacity = 2048U;

[[nodiscard]] std::string provider_status_text(int status) {
    switch (status) {
    case QUPY_PROVIDER_OK: return "ok";
    case QUPY_PROVIDER_BUFFER_TOO_SMALL: return "buffer too small";
    case QUPY_PROVIDER_INVALID_ARGUMENT: return "invalid argument";
    case QUPY_PROVIDER_UNAVAILABLE: return "unavailable";
    case QUPY_PROVIDER_REMOTE_ERROR: return "remote error";
    case QUPY_PROVIDER_INTERNAL_ERROR: return "internal error";
    default: return "unknown status " + std::to_string(status);
    }
}
class DynamicProviderLibrary {
public:
    explicit DynamicProviderLibrary(const std::string& path) {
#if defined(_WIN32)
        handle_ = LoadLibraryA(path.c_str());
#else
        handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
        if (handle_ == nullptr) {
            throw std::runtime_error("provider library could not be loaded");
        }
    }

    ~DynamicProviderLibrary() {
#if defined(_WIN32)
        if (handle_ != nullptr) FreeLibrary(handle_);
#else
        if (handle_ != nullptr) dlclose(handle_);
#endif
    }

    DynamicProviderLibrary(const DynamicProviderLibrary&) = delete;
    DynamicProviderLibrary& operator=(const DynamicProviderLibrary&) = delete;

    [[nodiscard]] std::uintptr_t symbol(const char* name) const noexcept {
#if defined(_WIN32)
        return reinterpret_cast<std::uintptr_t>(GetProcAddress(handle_, name));
#else
        return reinterpret_cast<std::uintptr_t>(dlsym(handle_, name));
#endif
    }

private:
#if defined(_WIN32)
    HMODULE handle_ = nullptr;
#else
    void* handle_ = nullptr;
#endif
};
template <typename Function>
[[nodiscard]] Function load_provider_symbol(
    const DynamicProviderLibrary& library,
    const char* name
) {
    const std::uintptr_t raw = library.symbol(name);
    if (raw == 0U) {
        throw std::runtime_error(std::string("provider library is missing symbol ") + name);
    }
    Function function{};
    static_assert(sizeof(function) == sizeof(raw));
    std::memcpy(&function, &raw, sizeof(function));
    return function;
}

struct TextBuffer {
    explicit TextBuffer(std::size_t capacity) : storage(capacity) {
        abi.data = storage.empty() ? nullptr : storage.data();
        abi.capacity = storage.size();
        abi.size = 0U;
    }

    [[nodiscard]] std::string text() const {
        if (abi.size > storage.size()) {
            throw std::runtime_error("provider returned a text size larger than its buffer");
        }
        return std::string(storage.data(), abi.size);
    }

    std::vector<char> storage;
    qupy_provider_buffer_v1 abi{};
};

[[nodiscard]] std::string provider_error(
    int status,
    const TextBuffer& error
) {
    const std::string detail = error.text();
    return detail.empty()
        ? "provider call failed: " + provider_status_text(status)
        : "provider call failed: " + provider_status_text(status) + ": " + detail;
}
template <typename Call>
[[nodiscard]] std::string read_provider_text(Call&& call) {
    qupy_provider_buffer_v1 probe{nullptr, 0U, 0U};
    const int probe_status = call(&probe);
    if (probe_status != QUPY_PROVIDER_OK && probe_status != QUPY_PROVIDER_BUFFER_TOO_SMALL) {
        throw std::runtime_error("provider text query failed: " + provider_status_text(probe_status));
    }
    if (probe.size == 0U) {
        return {};
    }
    TextBuffer output(probe.size);
    const int status = call(&output.abi);
    if (status != QUPY_PROVIDER_OK) {
        throw std::runtime_error("provider text query failed: " + provider_status_text(status));
    }
    return output.text();
}

[[nodiscard]] ProviderJobState provider_job_state(std::uint32_t state) {
    switch (state) {
    case QUPY_PROVIDER_JOB_QUEUED: return ProviderJobState::Queued;
    case QUPY_PROVIDER_JOB_RUNNING: return ProviderJobState::Running;
    case QUPY_PROVIDER_JOB_SUCCEEDED: return ProviderJobState::Succeeded;
    case QUPY_PROVIDER_JOB_FAILED: return ProviderJobState::Failed;
    case QUPY_PROVIDER_JOB_CANCELLED: return ProviderJobState::Cancelled;
    default: throw std::runtime_error("provider returned an unknown job state");
    }
}

}  // namespace

struct ProviderPlugin::Impl {
    explicit Impl(std::string path) : library(std::move(path)) {
        const auto get_provider = load_provider_symbol<qupy_provider_get_v1_fn>(
            library, "qupy_provider_get_v1"
        );
        provider = get_provider();
        if (provider == nullptr) {
            throw std::runtime_error("provider entry point returned null");
        }
        if (provider->abi_version != QUPY_PROVIDER_ABI_VERSION) {
            throw std::runtime_error("provider ABI version is not supported");
        }
        if (provider->provider_name == nullptr || *provider->provider_name == '\0') {
            throw std::runtime_error("provider name is empty");
        }
        if (provider->capabilities_json == nullptr || provider->submit == nullptr ||
            provider->poll == nullptr || provider->result_json == nullptr ||
            provider->cancel == nullptr) {
            throw std::runtime_error("provider is missing a required ABI function");
        }
        name = provider->provider_name;
    }

    ~Impl() {
        if (provider != nullptr && provider->destroy != nullptr) {
            provider->destroy(provider->context);
        }
    }

    DynamicProviderLibrary library;
    const qupy_provider_v1* provider = nullptr;
    std::string name;
};

ProviderPlugin::ProviderPlugin(std::string path)
    : impl_(std::make_shared<Impl>(std::move(path))) {}

const std::string& ProviderPlugin::name() const noexcept {
    return impl_->name;
}

std::string ProviderPlugin::capabilities_json() const {
    return read_provider_text([&](qupy_provider_buffer_v1* output) {
        return impl_->provider->capabilities_json(impl_->provider->context, output);
    });
}
std::string ProviderPlugin::submit(
    const ProviderProgram& program,
    std::uint64_t shots,
    const std::string& options_json
) const {
    if (shots == 0U) {
        throw std::invalid_argument("provider shot count must be positive");
    }
    const qupy_provider_submit_request_v1 request{
        program.format.c_str(),
        program.text.data(),
        program.text.size(),
        shots,
        options_json.data(),
        options_json.size(),
    };
    std::size_t job_capacity = kInitialJobIdCapacity;
    std::size_t error_capacity = kInitialErrorCapacity;
    for (int attempt = 0; attempt < 3; ++attempt) {
        TextBuffer job_id(job_capacity);
        TextBuffer error(error_capacity);
        const int status = impl_->provider->submit(
            impl_->provider->context, &request, &job_id.abi, &error.abi
        );
        if (status == QUPY_PROVIDER_BUFFER_TOO_SMALL) {
            job_capacity = std::max(job_capacity * 2U, job_id.abi.size);
            error_capacity = std::max(error_capacity * 2U, error.abi.size);
            continue;
        }
        if (status != QUPY_PROVIDER_OK) {
            throw std::runtime_error(provider_error(status, error));
        }
        const std::string result = job_id.text();
        if (result.empty()) {
            throw std::runtime_error("provider returned an empty job identifier");
        }
        return result;
    }
    throw std::runtime_error("provider submit buffers exceeded the supported retry limit");
}
ProviderJobState ProviderPlugin::poll(const std::string& job_id) const {
    if (job_id.empty()) {
        throw std::invalid_argument("provider job identifier must not be empty");
    }
    std::size_t error_capacity = kInitialErrorCapacity;
    for (int attempt = 0; attempt < 3; ++attempt) {
        TextBuffer error(error_capacity);
        std::uint32_t state = 0U;
        const int status = impl_->provider->poll(
            impl_->provider->context,
            job_id.data(),
            job_id.size(),
            &state,
            &error.abi
        );
        if (status == QUPY_PROVIDER_BUFFER_TOO_SMALL) {
            error_capacity = std::max(error_capacity * 2U, error.abi.size);
            continue;
        }
        if (status != QUPY_PROVIDER_OK) {
            throw std::runtime_error(provider_error(status, error));
        }
        return provider_job_state(state);
    }
    throw std::runtime_error("provider poll buffer exceeded the supported retry limit");
}

std::string ProviderPlugin::result_json(const std::string& job_id) const {
    if (job_id.empty()) {
        throw std::invalid_argument("provider job identifier must not be empty");
    }
    std::size_t output_capacity = 4096U;
    std::size_t error_capacity = kInitialErrorCapacity;
    for (int attempt = 0; attempt < 4; ++attempt) {
        TextBuffer output(output_capacity);
        TextBuffer error(error_capacity);
        const int status = impl_->provider->result_json(
            impl_->provider->context,
            job_id.data(),
            job_id.size(),
            &output.abi,
            &error.abi
        );
        if (status == QUPY_PROVIDER_BUFFER_TOO_SMALL) {
            output_capacity = std::max(output_capacity * 2U, output.abi.size);
            error_capacity = std::max(error_capacity * 2U, error.abi.size);
            continue;
        }
        if (status != QUPY_PROVIDER_OK) {
            throw std::runtime_error(provider_error(status, error));
        }
        return output.text();
    }
    throw std::runtime_error("provider result buffer exceeded the supported retry limit");
}

void ProviderPlugin::cancel(const std::string& job_id) const {
    if (job_id.empty()) {
        throw std::invalid_argument("provider job identifier must not be empty");
    }
    std::size_t error_capacity = kInitialErrorCapacity;
    for (int attempt = 0; attempt < 3; ++attempt) {
        TextBuffer error(error_capacity);
        const int status = impl_->provider->cancel(
            impl_->provider->context,
            job_id.data(),
            job_id.size(),
            &error.abi
        );
        if (status == QUPY_PROVIDER_BUFFER_TOO_SMALL) {
            error_capacity = std::max(error_capacity * 2U, error.abi.size);
            continue;
        }
        if (status != QUPY_PROVIDER_OK) {
            throw std::runtime_error(provider_error(status, error));
        }
        return;
    }
    throw std::runtime_error("provider cancel buffer exceeded the supported retry limit");
}

}  // namespace qupy
