#pragma once

#include "qupy/advanced.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace qupy {

struct BpOsdDecodeResult {
    std::vector<std::int8_t> correction;
    std::vector<std::int8_t> observables;
    double log_likelihood;
    std::size_t matched_errors;
    std::size_t iterations;
    bool bp_converged;
    bool osd_used;
};

struct BpOsdDecodeBatch {
    std::vector<std::int8_t> corrections;
    std::vector<std::int8_t> observables;
    std::vector<double> log_likelihoods;
    std::vector<std::uint64_t> matched_errors;
    std::vector<std::uint64_t> iterations;
    std::vector<std::int8_t> bp_converged;
    std::vector<std::int8_t> osd_used;
    std::size_t shots;
    std::size_t error_count;
    std::size_t observable_count;
};

class BpOsdDecoder {
public:
    explicit BpOsdDecoder(
        DetectorModel model,
        std::size_t max_iterations = 50U,
        double damping = 0.0
    );

    [[nodiscard]] const DetectorModel& model() const noexcept;
    [[nodiscard]] std::size_t max_iterations() const noexcept;
    [[nodiscard]] double damping() const noexcept;
    [[nodiscard]] std::size_t edge_count() const noexcept;
    [[nodiscard]] std::size_t active_error_count() const noexcept;

    [[nodiscard]] BpOsdDecodeResult decode(
        const std::vector<std::int8_t>& syndrome
    ) const;

    [[nodiscard]] BpOsdDecodeBatch decode_batch(
        const std::vector<std::int8_t>& syndromes,
        std::size_t shots
    ) const;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

[[nodiscard]] BpOsdDecodeResult decode_detector_model_bp_osd(
    const DetectorModel& model,
    const std::vector<std::int8_t>& syndrome,
    std::size_t max_iterations = 50U,
    double damping = 0.0
);

}  // namespace qupy
