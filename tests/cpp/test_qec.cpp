#include "qupy/qec.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

std::vector<std::int8_t> syndrome_from_correction(
    const qupy::DetectorModel& model,
    const std::vector<std::int8_t>& correction
) {
    assert(correction.size() == model.errors().size());
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
        assert(syndrome_from_correction(model, scalable.correction) == syndrome);
        assert(scalable.observables == exact.observables);
        assert(std::abs(scalable.log_likelihood - exact.log_likelihood) < 1e-12);
        assert(scalable.matched_errors == exact.matched_errors);
    }

    {
        const qupy::DetectorModel model = small_model();
        const std::vector<std::int8_t> syndrome{1, 1};
        const qupy::BpOsdDecoder decoder(model, 0U, 0.0);
        const qupy::BpOsdDecodeResult result = decoder.decode(syndrome);
        assert(result.osd_used);
        assert(!result.bp_converged);
        assert(result.iterations == 0U);
        assert(syndrome_from_correction(model, result.correction) == syndrome);
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
        assert((result.correction == std::vector<std::int8_t>{1, 1, 0}));
        assert((result.observables == std::vector<std::int8_t>{1}));
        assert(syndrome_from_correction(model, result.correction) ==
               std::vector<std::int8_t>({1, 1}));
        assert(decoder.active_error_count() == 1U);
        assert(decoder.edge_count() == 1U);
    }

    {
        const qupy::DetectorModel model(1U, 0U, {qupy::DetectorError{0.0, {0U}, {}}});
        bool rejected = false;
        try {
            (void) qupy::decode_detector_model_bp_osd(model, {1}, 0U, 0.0);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        assert(rejected);
    }

    {
        bool rejected = false;
        try {
            (void) qupy::BpOsdDecoder(small_model(), 10U, 1.0);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        assert(rejected);
    }

    {
        const qupy::DetectorModel model = large_sparse_model();
        assert(model.errors().size() == 64U);
        std::vector<std::int8_t> known(model.errors().size(), 0);
        known[3] = 1;
        known[17] = 1;
        known[31] = 1;
        const std::vector<std::int8_t> syndrome = syndrome_from_correction(model, known);

        bool exact_rejected = false;
        try {
            (void) qupy::decode_detector_model(model, syndrome);
        } catch (const std::invalid_argument&) {
            exact_rejected = true;
        }
        assert(exact_rejected);

        const qupy::BpOsdDecoder decoder(model, 30U, 0.1);
        const qupy::BpOsdDecodeResult result = decoder.decode(syndrome);
        assert(syndrome_from_correction(model, result.correction) == syndrome);
        assert(result.correction.size() == model.errors().size());
        assert(result.observables.size() == model.observable_count());

        const std::vector<std::int8_t> zero(model.detector_count(), 0);
        std::vector<std::int8_t> flattened;
        flattened.reserve(model.detector_count() * 3U);
        flattened.insert(flattened.end(), zero.begin(), zero.end());
        flattened.insert(flattened.end(), syndrome.begin(), syndrome.end());
        flattened.insert(flattened.end(), syndrome.begin(), syndrome.end());
        const qupy::BpOsdDecodeBatch batch = decoder.decode_batch(flattened, 3U);
        assert(batch.shots == 3U);
        assert(batch.error_count == model.errors().size());
        assert(batch.observable_count == model.observable_count());
        assert(batch.corrections.size() == 3U * model.errors().size());
        assert(batch.observables.size() == 3U * model.observable_count());
        assert(batch.log_likelihoods.size() == 3U);
        assert(batch.matched_errors.size() == 3U);
        assert(batch.iterations.size() == 3U);
        assert(batch.bp_converged.size() == 3U);
        assert(batch.osd_used.size() == 3U);

        std::vector<std::int8_t> batch_second(
            batch.corrections.begin() + static_cast<std::ptrdiff_t>(model.errors().size()),
            batch.corrections.begin() + static_cast<std::ptrdiff_t>(2U * model.errors().size())
        );
        assert(batch_second == result.correction);
    }

    return 0;
}
