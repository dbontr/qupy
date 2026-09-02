#include "qupy/provider_abi.h"

#include <cstring>
#include <string_view>

#if defined(_WIN32)
#define QUPY_TEST_EXPORT extern "C" __declspec(dllexport)
#else
#define QUPY_TEST_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

int write_text(std::string_view text, qupy_provider_buffer_v1* output) {
    if (output == nullptr) {
        return QUPY_PROVIDER_INVALID_ARGUMENT;
    }
    output->size = text.size();
    if (output->capacity < text.size() || output->data == nullptr) {
        return text.empty() ? QUPY_PROVIDER_OK : QUPY_PROVIDER_BUFFER_TOO_SMALL;
    }
    if (!text.empty()) {
        std::memcpy(output->data, text.data(), text.size());
    }
    return QUPY_PROVIDER_OK;
}

int capabilities(void*, qupy_provider_buffer_v1* output) {
    return write_text(R"({"formats":["openqasm3","qir-base-profile"]})", output);
}
int submit(
    void*,
    const qupy_provider_submit_request_v1* request,
    qupy_provider_buffer_v1* job_id,
    qupy_provider_buffer_v1* error
) {
    if (request == nullptr || request->program_format == nullptr || request->program_data == nullptr ||
        request->shots == 0U) {
        write_text("invalid submit request", error);
        return QUPY_PROVIDER_INVALID_ARGUMENT;
    }
    if (job_id == nullptr) {
        return QUPY_PROVIDER_INVALID_ARGUMENT;
    }
    constexpr std::string_view id = "fixture-job-1";
    job_id->size = id.size();
    if (job_id->data == nullptr || job_id->capacity < id.size()) {
        if (error != nullptr) error->size = 0U;
        return QUPY_PROVIDER_BUFFER_TOO_SMALL;
    }
    std::memcpy(job_id->data, id.data(), id.size());
    if (error != nullptr) error->size = 0U;
    return QUPY_PROVIDER_OK;
}

int poll(
    void*,
    const char*,
    size_t,
    uint32_t* state,
    qupy_provider_buffer_v1* error
) {
    if (state == nullptr) return QUPY_PROVIDER_INVALID_ARGUMENT;
    *state = QUPY_PROVIDER_JOB_SUCCEEDED;
    if (error != nullptr) error->size = 0U;
    return QUPY_PROVIDER_OK;
}
int result_json(
    void*,
    const char*,
    size_t,
    qupy_provider_buffer_v1* output,
    qupy_provider_buffer_v1* error
) {
    if (error != nullptr) error->size = 0U;
    return write_text(R"({"shots":32,"counts":{"00":16,"11":16}})", output);
}

int cancel(
    void*,
    const char*,
    size_t,
    qupy_provider_buffer_v1* error
) {
    if (error != nullptr) error->size = 0U;
    return QUPY_PROVIDER_OK;
}

const qupy_provider_v1 provider{
    QUPY_PROVIDER_ABI_VERSION,
    "qupy-test-provider",
    nullptr,
    &capabilities,
    &submit,
    &poll,
    &result_json,
    &cancel,
    nullptr,
};

}  // namespace

QUPY_TEST_EXPORT const qupy_provider_v1* qupy_provider_get_v1() {
    return &provider;
}
