#pragma once

#include <string>
#include <string_view>

namespace qupy::detail {

[[nodiscard]] std::string fingerprint_text(std::string_view text);

}  // namespace qupy::detail
