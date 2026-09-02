#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QUPY_PROVIDER_ABI_VERSION 1u

typedef enum qupy_provider_status_v1 {
    QUPY_PROVIDER_OK = 0,
    QUPY_PROVIDER_BUFFER_TOO_SMALL = 1,
    QUPY_PROVIDER_INVALID_ARGUMENT = 2,
    QUPY_PROVIDER_UNAVAILABLE = 3,
    QUPY_PROVIDER_REMOTE_ERROR = 4,
    QUPY_PROVIDER_INTERNAL_ERROR = 5
} qupy_provider_status_v1;

typedef enum qupy_provider_job_state_v1 {
    QUPY_PROVIDER_JOB_QUEUED = 0,
    QUPY_PROVIDER_JOB_RUNNING = 1,
    QUPY_PROVIDER_JOB_SUCCEEDED = 2,
    QUPY_PROVIDER_JOB_FAILED = 3,
    QUPY_PROVIDER_JOB_CANCELLED = 4
} qupy_provider_job_state_v1;
typedef struct qupy_provider_buffer_v1 {
    char* data;
    size_t capacity;
    size_t size;
} qupy_provider_buffer_v1;

typedef struct qupy_provider_submit_request_v1 {
    const char* program_format;
    const char* program_data;
    size_t program_size;
    uint64_t shots;
    const char* options_json;
    size_t options_json_size;
} qupy_provider_submit_request_v1;

typedef struct qupy_provider_v1 {
    uint32_t abi_version;
    const char* provider_name;
    void* context;

    int (*capabilities_json)(void* context, qupy_provider_buffer_v1* output);
    int (*submit)(
        void* context,
        const qupy_provider_submit_request_v1* request,
        qupy_provider_buffer_v1* job_id,
        qupy_provider_buffer_v1* error
    );
    int (*poll)(
        void* context,
        const char* job_id,
        size_t job_id_size,
        uint32_t* state,
        qupy_provider_buffer_v1* error
    );
    int (*result_json)(
        void* context,
        const char* job_id,
        size_t job_id_size,
        qupy_provider_buffer_v1* output,
        qupy_provider_buffer_v1* error
    );
    int (*cancel)(
        void* context,
        const char* job_id,
        size_t job_id_size,
        qupy_provider_buffer_v1* error
    );
    void (*destroy)(void* context);
} qupy_provider_v1;

typedef const qupy_provider_v1* (*qupy_provider_get_v1_fn)(void);

#ifdef __cplusplus
}
#endif
