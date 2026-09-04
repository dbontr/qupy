#include "qupy/qec.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<std::int8_t> syndrome_from_correction(
    const qupy::DetectorModel& model,
    const std::vector<std::int8_t>& correction
) {
    require(correction.size() == model.errors().size(), "correction size mismatch");
    std::vector<std::int8_t> syndrome(model.detector_count(), 0);
    for (std::size_t error_index = 0U; error_index < correction.size(); ++error_index) {
        if (correction[error_index] == 0) {
            continue;
        }
        for (const std::size_t detector : model.errors()[error_index].detectors) {
            syndrome[detector] ^= 1;
        }
    }
    return syndrome;
}

qupy::DetectorModel small_model() {
    return qupy::DetectorModel(
        2U,
        1U,
        {
            qupy::DetectorError{0.10, {0U}, {0U}},
            qupy::DetectorError{0.05, {1U}, {}},
            qupy::DetectorError{0.01, {0U, 1U}, {0U}},
        }
    );
}

qupy::DetectorModel large_sparse_model() {
    std::vector<qupy::DetectorError> errors;
    errors.reserve(64U);
    for (std::size_t detector = 0U; detector < 32U; ++detector) {
        errors.push_back({
            0.01 + 0.0001 * static_cast<double>(detector),
            {detector},
            detector == 31U ? std::vector<std::size_t>{0U} : std::vector<std::size_t>{},
        });
    }
    for (std::size_t detector = 0U; detector < 32U; ++detector) {
        errors.push_back({
            0.005 + 0.00005 * static_cast<double>(detector),
            {detector, (detector + 1U) % 32U},
            {},
        });
    }
    return qupy::DetectorModel(32U, 1U, std::move(errors));
}

}  // namespace

int main() {
    {
        const qupy::DetectorModel model = small_model();
        const std::vector<std::int8_t> syndrome{1, 0};
        const qupy::DecodeResult exact = qupy::decode_detector_model(model, syndrome);
        const qupy::BpOsdDecodeResult scalable =
            qupy::decode_detector_model_bp_osd(model, syndrome, 20U, 0.0);
        require(
            syndrome_from_correction(model, scalable.correction) == syndrome,
            "small-model correction has wrong syndrome"
        );
        require(scalable.observables == exact.observables, "logical frame differs from exact");
        require(
            std::abs(scalable.log_likelihood - exact.log_likelihood) < 1e-12,
            "small-model likelihood differs from exact"
        );
        require(scalable.matched_errors == exact.matched_errors, "matched-error count differs");
    }

    {
        const qupy::DetectorModel model = small_model();
        const std::vector<std::int8_t> syndrome{1, 1};
        const qupy::BpOsdDecoder decoder(model, 0U, 0.0);
        const qupy::BpOsdDecodeResult result = decoder.decode(syndrome);
        require(result.osd_used, "zero-iteration decode should use OSD");
        require(!result.bp_converged, "zero-iteration decode unexpectedly reports BP convergence");
        require(result.iterations == 0U, "zero-iteration decode reports BP iterations");
        require(
            syndrome_from_correction(model, result.correction) == syndrome,
            "OSD correction has wrong syndrome"
        );
    }

    {
        const qupy::DetectorModel model(
            2U,
            1U,
            {
                qupy::DetectorError{1.0, {0U}, {0U}},
                qupy::DetectorError{0.2, {1U}, {}},
                qupy::DetectorError{0.0, {0U, 1U}, {0U}},
            }
        );
        const qupy::BpOsdDecoder decoder(model, 10U, 0.25);
        const qupy::BpOsdDecodeResult result = decoder.decode({1, 1});
        require(
            result.correction == std::vector<std::int8_t>({1, 1, 0}),
            "deterministic endpoint correction mismatch"
        );
        require(
            result.observables == std::vector<std::int8_t>({1}),
            "deterministic endpoint logical frame mismatch"
        );
        require(
            syndrome_from_correction(model, result.correction) ==
                std::vector<std::int8_t>({1, 1}),
            "deterministic endpoint syndrome mismatch"
        );
        require(decoder.active_error_count() == 1U, "wrong active-error count");
        require(decoder.edge_count() == 1U, "wrong Tanner edge count");
    }

    {
        const qupy::DetectorModel model(1U, 0U, {qupy::DetectorError{0.0, {0U}, {}}});
        bool rejected = false;
        try {
            (void) qupy::decode_detector_model_bp_osd(model, {1}, 0U, 0.0);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "impossible syndrome was accepted");
    }

    {
        bool rejected = false;
        try {
            (void) qupy::BpOsdDecoder(small_model(), 10U, 1.0);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "invalid damping was accepted");
    }

    {
        const qupy::DetectorModel model = large_sparse_model();
        require(model.errors().size() == 64U, "large model did not contain 64 errors");
        std::vector<std::int8_t> known(model.errors().size(), 0);
        known[3] = 1;
        known[17] = 1;
        known[31] = 1;
        const std::vector<std::int8_t> syndrome = syndrome_from_correction(model, known);

        bool exact_rejected = false;
        try {
            (void) qupy::decode_detector_model(model, syndrome);
        } catch (const std::length_error&) {
            exact_rejected = true;
        }
        require(exact_rejected, "reference decoder unexpectedly accepted 64 error mechanisms");

        const qupy::BpOsdDecoder decoder(model, 30U, 0.1);
        const qupy::BpOsdDecodeResult result = decoder.decode(syndrome);
        require(
            syndrome_from_correction(model, result.correction) == syndrome,
            "large-model correction has wrong syndrome"
        );
        require(result.correction.size() == model.errors().size(), "wrong correction width");
        require(
            result.observables.size() == model.observable_count(),
            "wrong logical-frame width"
        );

        const std::vector<std::int8_t> zero(model.detector_count(), 0);
        std::vector<std::int8_t> flattened;
        flattened.reserve(model.detector_count() * 3U);
        flattened.insert(flattened.end(), zero.begin(), zero.end());
        flattened.insert(flattened.end(), syndrome.begin(), syndrome.end());
        flattened.insert(flattened.end(), syndrome.begin(), syndrome.end());
        const qupy::BpOsdDecodeBatch batch = decoder.decode_batch(flattened, 3U);
        require(batch.shots == 3U, "wrong batch shot count");
        require(batch.error_count == model.errors().size(), "wrong batch error width");
        require(
            batch.observable_count == model.observable_count(),
            "wrong batch observable width"
        );
        require(
            batch.corrections.size() == 3U * model.errors().size(),
            "wrong batch correction storage"
        );
        require(
            batch.observables.size() == 3U * model.observable_count(),
            "wrong batch observable storage"
        );
        require(batch.log_likelihoods.size() == 3U, "wrong likelihood batch size");
        require(batch.matched_errors.size() == 3U, "wrong matched-error batch size");
        require(batch.iterations.size() == 3U, "wrong iteration batch size");
        require(batch.bp_converged.size() == 3U, "wrong convergence batch size");
        require(batch.osd_used.size() == 3U, "wrong OSD batch size");

        const std::vector<std::int8_t> batch_second(
            batch.corrections.begin() + static_cast<std::ptrdiff_t>(model.errors().size()),
            batch.corrections.begin() + static_cast<std::ptrdiff_t>(2U * model.errors().size())
        );
        require(batch_second == result.correction, "batch decode differs from single decode");
    }

    return 0;
}
