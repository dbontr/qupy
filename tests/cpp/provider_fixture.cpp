#include "qupy/provider_abi.h"

#include <cstring>
#include <string_view>

#if defined(_WIN32)
#define QUPY_TEST_EXPORT extern "C" __declspec(dllexport)
#else
#define QUPY_TEST_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

struct FixtureContext {
    std::size_t polls = 0U;
    bool cancelled = false;
};

FixtureContext context;

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

bool fixture_job(std::string_view job_id) {
    return job_id == "fixture-job-1";
}

int capabilities(void*, qupy_provider_buffer_v1* output) {
    return write_text(R"({"formats":["openqasm3","qir-base-profile"]})", output);
}

int submit(
    void* raw_context,
    const qupy_provider_submit_request_v1* request,
    qupy_provider_buffer_v1* job_id,
    qupy_provider_buffer_v1* error
) {
    if (raw_context == nullptr || request == nullptr || request->program_format == nullptr ||
        request->program_data == nullptr || request->shots == 0U) {
        write_text("invalid submit request", error);
        return QUPY_PROVIDER_INVALID_ARGUMENT;
    }
    if (job_id == nullptr) {
        return QUPY_PROVIDER_INVALID_ARGUMENT;
    }
    auto& fixture = *static_cast<FixtureContext*>(raw_context);
    fixture.polls = 0U;
    fixture.cancelled = false;

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
    void* raw_context,
    const char* job_id,
    size_t job_id_size,
    uint32_t* state,
    qupy_provider_buffer_v1* error
) {
    if (raw_context == nullptr || job_id == nullptr || state == nullptr) {
        write_text("invalid poll request", error);
        return QUPY_PROVIDER_INVALID_ARGUMENT;
    }
    if (!fixture_job(std::string_view(job_id, job_id_size))) {
        write_text("unknown job", error);
        return QUPY_PROVIDER_REMOTE_ERROR;
    }

    auto& fixture = *static_cast<FixtureContext*>(raw_context);
    *state = fixture.cancelled ? QUPY_PROVIDER_JOB_CANCELLED : QUPY_PROVIDER_JOB_SUCCEEDED;
    ++fixture.polls;
    if (error != nullptr) error->size = 0U;
    return QUPY_PROVIDER_OK;
}

int result_json(
    void* raw_context,
    const char* job_id,
    size_t job_id_size,
    qupy_provider_buffer_v1* output,
    qupy_provider_buffer_v1* error
) {
    if (raw_context == nullptr || job_id == nullptr || output == nullptr) {
        write_text("invalid result request", error);
        return QUPY_PROVIDER_INVALID_ARGUMENT;
    }
    if (!fixture_job(std::string_view(job_id, job_id_size))) {
        write_text("unknown job", error);
        return QUPY_PROVIDER_REMOTE_ERROR;
    }

    const auto& fixture = *static_cast<const FixtureContext*>(raw_context);
    if (fixture.cancelled) {
        write_text("job cancelled", error);
        return QUPY_PROVIDER_REMOTE_ERROR;
    }
    if (fixture.polls == 0U) {
        write_text("job is not complete", error);
        return QUPY_PROVIDER_REMOTE_ERROR;
    }

    if (error != nullptr) error->size = 0U;
    return write_text(R"({"shots":32,"counts":{"00":16,"11":16}})", output);
}

int cancel(
    void* raw_context,
    const char* job_id,
    size_t job_id_size,
    qupy_provider_buffer_v1* error
) {
    if (raw_context == nullptr || job_id == nullptr) {
        write_text("invalid cancel request", error);
        return QUPY_PROVIDER_INVALID_ARGUMENT;
    }
    if (!fixture_job(std::string_view(job_id, job_id_size))) {
        write_text("unknown job", error);
        return QUPY_PROVIDER_REMOTE_ERROR;
    }

    auto& fixture = *static_cast<FixtureContext*>(raw_context);
    fixture.cancelled = true;
    if (error != nullptr) error->size = 0U;
    return QUPY_PROVIDER_OK;
}

const qupy_provider_v1 provider{
    QUPY_PROVIDER_ABI_VERSION,
    "qupy-test-provider",
    &context,
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
