#include "qupy/qec.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qupy {
namespace {

constexpr double kMessageLimit = 50.0;
constexpr double kTanhProductLimit = 1.0 - 1e-15;

[[nodiscard]] std::vector<std::size_t> parity_support(
    const std::vector<std::size_t>& values
) {
    std::vector<std::size_t> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    std::vector<std::size_t> result;
    for (std::size_t begin = 0U; begin < sorted.size();) {
        std::size_t end = begin + 1U;
        while (end < sorted.size() && sorted[end] == sorted[begin]) {
            ++end;
        }
        if (((end - begin) & 1U) != 0U) {
            result.push_back(sorted[begin]);
        }
        begin = end;
    }
    return result;
}

void validate_syndrome(
    const std::vector<std::int8_t>& syndrome,
    std::size_t detector_count
) {
    if (syndrome.size() != detector_count) {
        throw std::invalid_argument("syndrome length must match detector count");
    }
    for (const std::int8_t value : syndrome) {
        if (value != 0 && value != 1) {
            throw std::invalid_argument("syndrome values must be zero or one");
        }
    }
}

[[nodiscard]] std::size_t checked_product(
    std::size_t left,
    std::size_t right,
    const char* label
) {
    if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::length_error(std::string(label) + " exceeds native range");
    }
    return left * right;
}

[[nodiscard]] double clamp_message(double value) noexcept {
    return std::clamp(value, -kMessageLimit, kMessageLimit);
}

[[nodiscard]] std::size_t word_count(std::size_t bit_count) noexcept {
    return (bit_count + 63U) / 64U;
}

void set_bit(std::vector<std::uint64_t>& words, std::size_t bit_index) {
    words[bit_index / 64U] |= std::uint64_t{1} << (bit_index % 64U);
}

[[nodiscard]] bool get_bit(
    const std::vector<std::uint64_t>& words,
    std::size_t bit_index
) noexcept {
    return (words[bit_index / 64U] &
            (std::uint64_t{1} << (bit_index % 64U))) != 0U;
}

void xor_words(
    std::vector<std::uint64_t>& destination,
    const std::vector<std::uint64_t>& source
) {
    for (std::size_t index = 0U; index < destination.size(); ++index) {
        destination[index] ^= source[index];
    }
}

[[nodiscard]] std::size_t first_set_bit(
    const std::vector<std::uint64_t>& words,
    std::size_t bit_count
) noexcept {
    for (std::size_t word_index = 0U; word_index < words.size(); ++word_index) {
        const std::uint64_t word = words[word_index];
        if (word == 0U) {
            continue;
        }
        const std::size_t bit = word_index * 64U + std::countr_zero(word);
        return bit < bit_count ? bit : bit_count;
    }
    return bit_count;
}

void toggle_support(
    std::vector<std::int8_t>& bits,
    const std::vector<std::size_t>& support
) {
    for (const std::size_t index : support) {
        bits[index] ^= 1;
    }
}

[[nodiscard]] bool syndrome_matches(
    const std::vector<std::vector<std::size_t>>& supports,
    const std::vector<std::int8_t>& assignment,
    const std::vector<std::int8_t>& target
) {
    std::vector<std::int8_t> actual(target.size(), 0);
    for (std::size_t variable = 0U; variable < assignment.size(); ++variable) {
        if (assignment[variable] != 0) {
            toggle_support(actual, supports[variable]);
        }
    }
    return actual == target;
}

}  // namespace

struct BpOsdDecoder::Impl {
    struct ActiveError {
        std::size_t original_index;
        double probability;
        double prior_llr;
        std::vector<std::size_t> detectors;
        std::vector<std::size_t> observables;
        std::vector<std::size_t> edges;
    };

    struct Edge {
        std::size_t detector;
        std::size_t variable;
    };

    DetectorModel model;
    std::size_t max_iterations;
    double damping;
    std::vector<ActiveError> active_errors;
    std::vector<Edge> edges;
    std::vector<std::vector<std::size_t>> check_edges;
    std::vector<std::int8_t> fixed_correction;
    std::vector<std::int8_t> fixed_syndrome;

    Impl(DetectorModel input, std::size_t iterations, double damping_value)
        : model(std::move(input)),
          max_iterations(iterations),
          damping(damping_value),
          check_edges(model.detector_count()),
          fixed_correction(model.errors().size(), 0),
          fixed_syndrome(model.detector_count(), 0) {
        if (!std::isfinite(damping) || damping < 0.0 || damping >= 1.0) {
            throw std::invalid_argument("BP damping must be finite and in [0, 1)");
        }

        for (std::size_t original = 0U; original < model.errors().size(); ++original) {
            const DetectorError& error = model.errors()[original];
            const std::vector<std::size_t> detectors = parity_support(error.detectors);
            const std::vector<std::size_t> observables = parity_support(error.observables);
            if (error.probability == 0.0) {
                continue;
            }
            if (error.probability == 1.0) {
                fixed_correction[original] = 1;
                toggle_support(fixed_syndrome, detectors);
                continue;
            }
            const double prior_llr =
                std::log1p(-error.probability) - std::log(error.probability);
            if (!std::isfinite(prior_llr)) {
                throw std::invalid_argument(
                    "detector error probability produced a non-finite BP prior"
                );
            }
            active_errors.push_back({
                original,
                error.probability,
                prior_llr,
                detectors,
                observables,
                {},
            });
        }

        for (std::size_t variable = 0U; variable < active_errors.size(); ++variable) {
            ActiveError& error = active_errors[variable];
            error.edges.reserve(error.detectors.size());
            for (const std::size_t detector : error.detectors) {
                const std::size_t edge_index = edges.size();
                edges.push_back({detector, variable});
                error.edges.push_back(edge_index);
                check_edges[detector].push_back(edge_index);
            }
        }
    }

    [[nodiscard]] std::vector<std::int8_t> adjusted_syndrome(
        const std::vector<std::int8_t>& syndrome
    ) const {
        validate_syndrome(syndrome, model.detector_count());
        std::vector<std::int8_t> target = syndrome;
        for (std::size_t detector = 0U; detector < target.size(); ++detector) {
            target[detector] ^= fixed_syndrome[detector];
        }
        return target;
    }

    [[nodiscard]] std::vector<std::vector<std::size_t>> supports() const {
        std::vector<std::vector<std::size_t>> result;
        result.reserve(active_errors.size());
        for (const ActiveError& error : active_errors) {
            result.push_back(error.detectors);
        }
        return result;
    }

    [[nodiscard]] std::vector<std::size_t> osd_pivots(
        const std::vector<double>& posterior_llr
    ) const {
        std::vector<std::size_t> order(active_errors.size());
        std::iota(order.begin(), order.end(), 0U);
        std::sort(
            order.begin(),
            order.end(),
            [&](std::size_t left, std::size_t right) {
                const double left_reliability = std::abs(posterior_llr[left]);
                const double right_reliability = std::abs(posterior_llr[right]);
                if (left_reliability != right_reliability) {
                    return left_reliability < right_reliability;
                }
                return active_errors[left].original_index <
                       active_errors[right].original_index;
            }
        );

        const std::size_t detector_count = model.detector_count();
        const std::size_t words = word_count(detector_count);
        std::vector<std::vector<std::uint64_t>> basis(detector_count);
        std::vector<std::int8_t> present(detector_count, 0);
        std::vector<std::size_t> selected;
        selected.reserve(std::min(detector_count, active_errors.size()));

        for (const std::size_t variable : order) {
            std::vector<std::uint64_t> column(words, 0U);
            for (const std::size_t detector : active_errors[variable].detectors) {
                set_bit(column, detector);
            }
            for (std::size_t pivot = 0U; pivot < detector_count; ++pivot) {
                if (present[pivot] != 0 && get_bit(column, pivot)) {
                    xor_words(column, basis[pivot]);
                }
            }
            const std::size_t pivot = first_set_bit(column, detector_count);
            if (pivot == detector_count) {
                continue;
            }
            basis[pivot] = std::move(column);
            present[pivot] = 1;
            selected.push_back(variable);
        }
        return selected;
    }

    [[nodiscard]] std::vector<std::int8_t> osd0(
        const std::vector<std::int8_t>& target,
        const std::vector<std::int8_t>& hard,
        const std::vector<double>& posterior_llr
    ) const {
        const std::vector<std::size_t> pivots = osd_pivots(posterior_llr);
        std::vector<std::int8_t> is_pivot(active_errors.size(), 0);
        for (const std::size_t variable : pivots) {
            is_pivot[variable] = 1;
        }

        std::vector<std::int8_t> rhs = target;
        for (std::size_t variable = 0U; variable < active_errors.size(); ++variable) {
            if (is_pivot[variable] == 0 && hard[variable] != 0) {
                toggle_support(rhs, active_errors[variable].detectors);
            }
        }

        const std::size_t column_count = pivots.size();
        const std::size_t rhs_bit = column_count;
        const std::size_t row_words = word_count(column_count + 1U);
        std::vector<std::vector<std::uint64_t>> rows(
            model.detector_count(),
            std::vector<std::uint64_t>(row_words, 0U)
        );
        for (std::size_t column = 0U; column < column_count; ++column) {
            for (const std::size_t detector : active_errors[pivots[column]].detectors) {
                set_bit(rows[detector], column);
            }
        }
        for (std::size_t detector = 0U; detector < rhs.size(); ++detector) {
            if (rhs[detector] != 0) {
                set_bit(rows[detector], rhs_bit);
            }
        }

        std::vector<std::size_t> pivot_rows(column_count, 0U);
        std::size_t next_row = 0U;
        for (std::size_t column = 0U; column < column_count; ++column) {
            std::size_t row = next_row;
            while (row < rows.size() && !get_bit(rows[row], column)) {
                ++row;
            }
            if (row == rows.size()) {
                throw std::logic_error("OSD pivot set lost linear independence");
            }
            std::swap(rows[next_row], rows[row]);
            for (std::size_t other = 0U; other < rows.size(); ++other) {
                if (other != next_row && get_bit(rows[other], column)) {
                    xor_words(rows[other], rows[next_row]);
                }
            }
            pivot_rows[column] = next_row;
            ++next_row;
        }

        for (std::size_t row = next_row; row < rows.size(); ++row) {
            if (get_bit(rows[row], rhs_bit)) {
                throw std::invalid_argument(
                    "syndrome is not representable by nonzero-probability detector errors"
                );
            }
        }

        std::vector<std::int8_t> corrected = hard;
        for (std::size_t column = 0U; column < column_count; ++column) {
            corrected[pivots[column]] =
                get_bit(rows[pivot_rows[column]], rhs_bit) ? 1 : 0;
        }
        return corrected;
    }

    [[nodiscard]] BpOsdDecodeResult decode(
        const std::vector<std::int8_t>& syndrome
    ) const {
        const std::vector<std::int8_t> target = adjusted_syndrome(syndrome);
        const std::vector<std::vector<std::size_t>> variable_supports = supports();

        std::vector<double> check_to_variable(edges.size(), 0.0);
        std::vector<double> variable_to_check(edges.size(), 0.0);
        std::vector<double> posterior(active_errors.size(), 0.0);
        std::vector<std::int8_t> hard(active_errors.size(), 0);
        for (std::size_t variable = 0U; variable < active_errors.size(); ++variable) {
            posterior[variable] = active_errors[variable].prior_llr;
            hard[variable] = posterior[variable] < 0.0 ? 1 : 0;
            for (const std::size_t edge : active_errors[variable].edges) {
                variable_to_check[edge] = clamp_message(posterior[variable]);
            }
        }

        bool converged = syndrome_matches(variable_supports, hard, target);
        std::size_t iterations = 0U;
        for (std::size_t iteration = 1U;
             !converged && iteration <= max_iterations;
             ++iteration) {
            std::vector<double> next_check_to_variable(edges.size(), 0.0);
            for (std::size_t detector = 0U; detector < check_edges.size(); ++detector) {
                const auto& incident = check_edges[detector];
                for (const std::size_t edge : incident) {
                    double product = target[detector] != 0 ? -1.0 : 1.0;
                    for (const std::size_t other : incident) {
                        if (other == edge) {
                            continue;
                        }
                        product *= std::tanh(clamp_message(variable_to_check[other]) / 2.0);
                    }
                    product = std::clamp(
                        product,
                        -kTanhProductLimit,
                        kTanhProductLimit
                    );
                    const double update = clamp_message(2.0 * std::atanh(product));
                    next_check_to_variable[edge] = clamp_message(
                        damping * check_to_variable[edge] +
                        (1.0 - damping) * update
                    );
                }
            }
            check_to_variable.swap(next_check_to_variable);

            for (std::size_t variable = 0U; variable < active_errors.size(); ++variable) {
                double total = active_errors[variable].prior_llr;
                for (const std::size_t edge : active_errors[variable].edges) {
                    total += check_to_variable[edge];
                }
                if (!std::isfinite(total)) {
                    throw std::overflow_error("BP produced a non-finite posterior LLR");
                }
                posterior[variable] = total;
                hard[variable] = total < 0.0 ? 1 : 0;
                for (const std::size_t edge : active_errors[variable].edges) {
                    variable_to_check[edge] = clamp_message(
                        total - check_to_variable[edge]
                    );
                }
            }
            iterations = iteration;
            converged = syndrome_matches(variable_supports, hard, target);
        }

        const bool osd_used = !converged;
        const std::vector<std::int8_t> active_correction =
            osd_used ? osd0(target, hard, posterior) : hard;
        if (!syndrome_matches(variable_supports, active_correction, target)) {
            throw std::logic_error("BP-OSD returned a correction with the wrong syndrome");
        }

        std::vector<std::int8_t> correction = fixed_correction;
        for (std::size_t variable = 0U; variable < active_errors.size(); ++variable) {
            correction[active_errors[variable].original_index] = active_correction[variable];
        }

        std::vector<std::int8_t> check(model.detector_count(), 0);
        std::vector<std::int8_t> observables(model.observable_count(), 0);
        double log_likelihood = 0.0;
        std::size_t matched_errors = 0U;
        for (std::size_t index = 0U; index < model.errors().size(); ++index) {
            const DetectorError& error = model.errors()[index];
            const bool selected = correction[index] != 0;
            if (selected) {
                ++matched_errors;
                for (const std::size_t detector : error.detectors) {
                    check[detector] ^= 1;
                }
                for (const std::size_t observable : error.observables) {
                    observables[observable] ^= 1;
                }
            }
            if (selected) {
                if (error.probability == 0.0) {
                    throw std::logic_error("BP-OSD selected an impossible detector error");
                }
                log_likelihood += error.probability == 1.0
                    ? 0.0
                    : std::log(error.probability);
            } else {
                if (error.probability == 1.0) {
                    throw std::logic_error("BP-OSD omitted a certain detector error");
                }
                log_likelihood += error.probability == 0.0
                    ? 0.0
                    : std::log1p(-error.probability);
            }
        }
        if (check != syndrome) {
            throw std::logic_error("BP-OSD full correction failed syndrome verification");
        }

        return {
            std::move(correction),
            std::move(observables),
            log_likelihood,
            matched_errors,
            iterations,
            converged,
            osd_used,
        };
    }
};

BpOsdDecoder::BpOsdDecoder(
    DetectorModel model,
    std::size_t max_iterations,
    double damping
) : impl_(std::make_shared<Impl>(std::move(model), max_iterations, damping)) {}

const DetectorModel& BpOsdDecoder::model() const noexcept {
    return impl_->model;
}

std::size_t BpOsdDecoder::max_iterations() const noexcept {
    return impl_->max_iterations;
}

double BpOsdDecoder::damping() const noexcept {
    return impl_->damping;
}

std::size_t BpOsdDecoder::edge_count() const noexcept {
    return impl_->edges.size();
}

std::size_t BpOsdDecoder::active_error_count() const noexcept {
    return impl_->active_errors.size();
}

BpOsdDecodeResult BpOsdDecoder::decode(
    const std::vector<std::int8_t>& syndrome
) const {
    return impl_->decode(syndrome);
}

BpOsdDecodeBatch BpOsdDecoder::decode_batch(
    const std::vector<std::int8_t>& syndromes,
    std::size_t shots
) const {
    const std::size_t detector_count = impl_->model.detector_count();
    if (syndromes.size() != checked_product(shots, detector_count, "syndrome batch")) {
        throw std::invalid_argument(
            "syndrome batch size must equal shots times detector count"
        );
    }

    BpOsdDecodeBatch batch;
    batch.shots = shots;
    batch.error_count = impl_->model.errors().size();
    batch.observable_count = impl_->model.observable_count();
    batch.corrections.resize(
        checked_product(shots, batch.error_count, "BP-OSD correction batch")
    );
    batch.observables.resize(
        checked_product(shots, batch.observable_count, "BP-OSD observable batch")
    );
    batch.log_likelihoods.resize(shots);
    batch.matched_errors.resize(shots);
    batch.iterations.resize(shots);
    batch.bp_converged.resize(shots);
    batch.osd_used.resize(shots);

    for (std::size_t shot = 0U; shot < shots; ++shot) {
        const auto begin = syndromes.begin() + static_cast<std::ptrdiff_t>(shot * detector_count);
        const std::vector<std::int8_t> syndrome(
            begin,
            begin + static_cast<std::ptrdiff_t>(detector_count)
        );
        BpOsdDecodeResult result = impl_->decode(syndrome);
        std::copy(
            result.correction.begin(),
            result.correction.end(),
            batch.corrections.begin() + static_cast<std::ptrdiff_t>(shot * batch.error_count)
        );
        std::copy(
            result.observables.begin(),
            result.observables.end(),
            batch.observables.begin() +
                static_cast<std::ptrdiff_t>(shot * batch.observable_count)
        );
        batch.log_likelihoods[shot] = result.log_likelihood;
        batch.matched_errors[shot] = static_cast<std::uint64_t>(result.matched_errors);
        batch.iterations[shot] = static_cast<std::uint64_t>(result.iterations);
        batch.bp_converged[shot] = result.bp_converged ? 1 : 0;
        batch.osd_used[shot] = result.osd_used ? 1 : 0;
    }
    return batch;
}

BpOsdDecodeResult decode_detector_model_bp_osd(
    const DetectorModel& model,
    const std::vector<std::int8_t>& syndrome,
    std::size_t max_iterations,
    double damping
) {
    return BpOsdDecoder(model, max_iterations, damping).decode(syndrome);
}

}  // namespace qupy
