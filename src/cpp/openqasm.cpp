#include "qupy/circuit.hpp"

#include <cctype>
#include <cmath>
#include <cstddef>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace qupy {
namespace {

enum class TokenKind {
    Identifier,
    Number,
    String,
    LBracket,
    RBracket,
    LParen,
    RParen,
    LBrace,
    RBrace,
    Comma,
    Semicolon,
    Assign,
    EqualEqual,
    End,
};

struct Token {
    TokenKind kind;
    std::string text;
    std::size_t line;
    std::size_t column;
};

[[noreturn]] void parse_error(
    std::size_t line,
    std::size_t column,
    const std::string& message
) {
    throw std::invalid_argument(
        "OpenQASM parse error at line " + std::to_string(line) +
        ", column " + std::to_string(column) + ": " + message
    );
}

class Lexer {
public:
    explicit Lexer(std::string_view input) : input_(input) {}

    [[nodiscard]] std::vector<Token> tokenize() {
        std::vector<Token> tokens;
        while (true) {
            skip_trivia();
            if (position_ == input_.size()) {
                tokens.push_back({TokenKind::End, {}, line_, column_});
                return tokens;
            }
            tokens.push_back(next_token());
        }
    }

private:
    [[nodiscard]] char peek(std::size_t offset = 0U) const noexcept {
        const std::size_t index = position_ + offset;
        return index < input_.size() ? input_[index] : '\0';
    }

    char advance() noexcept {
        const char value = input_[position_++];
        if (value == '\n') {
            ++line_;
            column_ = 1U;
        } else {
            ++column_;
        }
        return value;
    }

    void skip_trivia() {
        while (position_ < input_.size()) {
            if (std::isspace(static_cast<unsigned char>(peek())) != 0) {
                advance();
                continue;
            }
            if (peek() == '/' && peek(1U) == '/') {
                advance();
                advance();
                while (position_ < input_.size() && peek() != '\n') {
                    advance();
                }
                continue;
            }
            if (peek() == '/' && peek(1U) == '*') {
                const std::size_t start_line = line_;
                const std::size_t start_column = column_;
                advance();
                advance();
                bool closed = false;
                while (position_ < input_.size()) {
                    if (peek() == '*' && peek(1U) == '/') {
                        advance();
                        advance();
                        closed = true;
                        break;
                    }
                    advance();
                }
                if (!closed) {
                    parse_error(start_line, start_column, "unterminated block comment");
                }
                continue;
            }
            return;
        }
    }

    [[nodiscard]] static bool identifier_start(char value) noexcept {
        const unsigned char converted = static_cast<unsigned char>(value);
        return std::isalpha(converted) != 0 || value == '_';
    }

    [[nodiscard]] static bool identifier_continue(char value) noexcept {
        const unsigned char converted = static_cast<unsigned char>(value);
        return std::isalnum(converted) != 0 || value == '_';
    }

    [[nodiscard]] bool number_start() const noexcept {
        const char first = peek();
        const char second = peek(1U);
        if (std::isdigit(static_cast<unsigned char>(first)) != 0) {
            return true;
        }
        if (first == '.' && std::isdigit(static_cast<unsigned char>(second)) != 0) {
            return true;
        }
        return (first == '+' || first == '-') &&
            (std::isdigit(static_cast<unsigned char>(second)) != 0 || second == '.');
    }

    [[nodiscard]] Token identifier_token(
        std::size_t start,
        std::size_t line,
        std::size_t column
    ) {
        advance();
        while (identifier_continue(peek())) {
            advance();
        }
        return {
            TokenKind::Identifier,
            std::string(input_.substr(start, position_ - start)),
            line,
            column,
        };
    }

    [[nodiscard]] Token number_token(
        std::size_t start,
        std::size_t line,
        std::size_t column
    ) {
        if (peek() == '+' || peek() == '-') {
            advance();
        }
        bool digits = false;
        while (std::isdigit(static_cast<unsigned char>(peek())) != 0) {
            advance();
            digits = true;
        }
        if (peek() == '.') {
            advance();
            while (std::isdigit(static_cast<unsigned char>(peek())) != 0) {
                advance();
                digits = true;
            }
        }
        if (!digits) {
            parse_error(line, column, "invalid numeric literal");
        }
        if (peek() == 'e' || peek() == 'E') {
            advance();
            if (peek() == '+' || peek() == '-') {
                advance();
            }
            bool exponent_digits = false;
            while (std::isdigit(static_cast<unsigned char>(peek())) != 0) {
                advance();
                exponent_digits = true;
            }
            if (!exponent_digits) {
                parse_error(line, column, "invalid numeric exponent");
            }
        }
        return {
            TokenKind::Number,
            std::string(input_.substr(start, position_ - start)),
            line,
            column,
        };
    }

    [[nodiscard]] Token string_token(std::size_t line, std::size_t column) {
        advance();
        std::string value;
        while (position_ < input_.size()) {
            const char current = advance();
            if (current == '"') {
                return {TokenKind::String, std::move(value), line, column};
            }
            if (current == '\n' || current == '\r') {
                parse_error(line, column, "unterminated string literal");
            }
            if (current == '\\') {
                if (position_ == input_.size()) {
                    parse_error(line, column, "unterminated string escape");
                }
                value.push_back(advance());
            } else {
                value.push_back(current);
            }
        }
        parse_error(line, column, "unterminated string literal");
    }

    [[nodiscard]] Token next_token() {
        const std::size_t start = position_;
        const std::size_t token_line = line_;
        const std::size_t token_column = column_;
        const char current = peek();
        if (identifier_start(current)) {
            return identifier_token(start, token_line, token_column);
        }
        if (number_start()) {
            return number_token(start, token_line, token_column);
        }
        if (current == '"') {
            return string_token(token_line, token_column);
        }

        advance();
        switch (current) {
        case '[': return {TokenKind::LBracket, "[", token_line, token_column};
        case ']': return {TokenKind::RBracket, "]", token_line, token_column};
        case '(': return {TokenKind::LParen, "(", token_line, token_column};
        case ')': return {TokenKind::RParen, ")", token_line, token_column};
        case '{': return {TokenKind::LBrace, "{", token_line, token_column};
        case '}': return {TokenKind::RBrace, "}", token_line, token_column};
        case ',': return {TokenKind::Comma, ",", token_line, token_column};
        case ';': return {TokenKind::Semicolon, ";", token_line, token_column};
        case '=':
            if (peek() == '=') {
                advance();
                return {TokenKind::EqualEqual, "==", token_line, token_column};
            }
            return {TokenKind::Assign, "=", token_line, token_column};
        default:
            parse_error(
                token_line,
                token_column,
                std::string("unsupported character '") + current + "'"
            );
        }
    }

    std::string_view input_;
    std::size_t position_ = 0U;
    std::size_t line_ = 1U;
    std::size_t column_ = 1U;
};

class Parser {
public:
    explicit Parser(std::string_view input) : tokens_(Lexer(input).tokenize()) {}

    [[nodiscard]] Circuit parse() {
        expect_identifier("OPENQASM");
        const Token& version = expect(TokenKind::Number, "OpenQASM version");
        if (version.text != "3" && version.text != "3.0" && version.text != "3.1") {
            fail(version, "only OpenQASM 3 and 3.1 are supported");
        }
        expect(TokenKind::Semicolon, "';' after OpenQASM version");

        bool saw_stdgates = false;
        bool saw_qubits = false;
        bool saw_clbits = false;
        std::size_t num_qubits = 0U;
        std::size_t num_clbits = 0U;
        std::string qubit_register;
        std::string clbit_register;

        while (peek().kind == TokenKind::Identifier) {
            if (peek().text == "include") {
                consume();
                const Token& include = expect(TokenKind::String, "include path");
                if (include.text != "stdgates.inc") {
                    fail(include, "only stdgates.inc is supported");
                }
                if (saw_stdgates) {
                    fail(include, "stdgates.inc was included more than once");
                }
                saw_stdgates = true;
                expect(TokenKind::Semicolon, "';' after include");
                continue;
            }
            if (peek().text == "qubit") {
                if (saw_qubits) {
                    fail(peek(), "multiple quantum registers are not supported");
                }
                consume();
                expect(TokenKind::LBracket, "'[' after qubit");
                const Token& size = expect(TokenKind::Number, "quantum register size");
                num_qubits = parse_size(size);
                if (num_qubits == 0U) {
                    fail(size, "quantum register size must be positive");
                }
                expect(TokenKind::RBracket, "']' after quantum register size");
                qubit_register = expect(
                    TokenKind::Identifier,
                    "quantum register name"
                ).text;
                expect(TokenKind::Semicolon, "';' after quantum register declaration");
                saw_qubits = true;
                continue;
            }
            if (peek().text == "bit") {
                if (saw_clbits) {
                    fail(peek(), "multiple classical registers are not supported");
                }
                consume();
                expect(TokenKind::LBracket, "'[' after bit");
                const Token& size = expect(TokenKind::Number, "classical register size");
                num_clbits = parse_size(size);
                if (num_clbits == 0U) {
                    fail(size, "classical register size must be positive");
                }
                expect(TokenKind::RBracket, "']' after classical register size");
                clbit_register = expect(
                    TokenKind::Identifier,
                    "classical register name"
                ).text;
                expect(TokenKind::Semicolon, "';' after classical register declaration");
                saw_clbits = true;
                continue;
            }
            break;
        }

        if (!saw_qubits) {
            fail(peek(), "exactly one quantum register declaration is required");
        }
        if (saw_clbits && clbit_register == qubit_register) {
            fail(peek(), "quantum and classical registers must have distinct names");
        }

        qubit_register_ = std::move(qubit_register);
        clbit_register_ = std::move(clbit_register);
        num_qubits_ = num_qubits;
        num_clbits_ = num_clbits;

        Circuit circuit(num_qubits_, num_clbits_);
        while (peek().kind != TokenKind::End) {
            circuit = parse_statement(std::move(circuit), std::nullopt);
        }
        return circuit;
    }

private:
    [[noreturn]] void fail(const Token& token, const std::string& message) const {
        parse_error(token.line, token.column, message);
    }

    [[nodiscard]] const Token& peek(std::size_t offset = 0U) const {
        const std::size_t index = position_ + offset;
        return tokens_[index < tokens_.size() ? index : tokens_.size() - 1U];
    }

    const Token& consume() {
        return tokens_[position_++];
    }

    const Token& expect(TokenKind kind, const std::string& description) {
        if (peek().kind != kind) {
            fail(peek(), "expected " + description);
        }
        return consume();
    }

    const Token& expect_identifier(std::string_view value) {
        if (peek().kind != TokenKind::Identifier || peek().text != value) {
            fail(peek(), "expected '" + std::string(value) + "'");
        }
        return consume();
    }

    [[nodiscard]] std::size_t parse_size(const Token& token) const {
        if (token.text.empty()) {
            fail(token, "expected a non-negative integer");
        }
        std::size_t value = 0U;
        for (const char digit : token.text) {
            if (std::isdigit(static_cast<unsigned char>(digit)) == 0) {
                fail(token, "expected a non-negative integer");
            }
            const std::size_t numeric = static_cast<std::size_t>(digit - '0');
            if (value > (std::numeric_limits<std::size_t>::max() - numeric) / 10U) {
                fail(token, "integer is outside the native size range");
            }
            value = value * 10U + numeric;
        }
        return value;
    }

    [[nodiscard]] double parse_double(const Token& token) const {
        std::istringstream input(token.text);
        input.imbue(std::locale::classic());
        double value = 0.0;
        input >> value;
        if (!input || !input.eof() || !std::isfinite(value)) {
            fail(token, "rotation parameter must be a finite numeric literal");
        }
        return value;
    }

    [[nodiscard]] std::size_t parse_register_ref(
        const std::string& register_name,
        std::size_t register_size,
        const std::string& kind
    ) {
        const Token& name = expect(TokenKind::Identifier, kind + " register name");
        if (register_name.empty() || name.text != register_name) {
            fail(name, "unknown " + kind + " register '" + name.text + "'");
        }
        expect(TokenKind::LBracket, "'[' after " + kind + " register");
        const Token& index_token = expect(TokenKind::Number, kind + " index");
        const std::size_t index = parse_size(index_token);
        expect(TokenKind::RBracket, "']' after " + kind + " index");
        if (index >= register_size) {
            fail(index_token, kind + " index is outside the declared register");
        }
        return index;
    }

    [[nodiscard]] std::size_t parse_qubit_ref() {
        return parse_register_ref(qubit_register_, num_qubits_, "quantum");
    }

    [[nodiscard]] std::size_t parse_clbit_ref() {
        if (num_clbits_ == 0U) {
            fail(peek(), "circuit has no classical register");
        }
        return parse_register_ref(clbit_register_, num_clbits_, "classical");
    }

    [[nodiscard]] Circuit parse_if(Circuit circuit) {
        expect_identifier("if");
        expect(TokenKind::LParen, "'(' after if");
        const std::size_t bit = parse_clbit_ref();
        expect(TokenKind::EqualEqual, "'==' in classical condition");
        const Token& value = expect(TokenKind::Number, "condition value");
        if (value.text != "0" && value.text != "1") {
            fail(value, "condition value must be 0 or 1");
        }
        expect(TokenKind::RParen, "')' after classical condition");
        expect(TokenKind::LBrace, "'{' after classical condition");
        if (peek().kind == TokenKind::RBrace) {
            fail(peek(), "conditional block must contain exactly one instruction");
        }
        circuit = parse_statement(
            std::move(circuit),
            ClassicalCondition{bit, value.text == "1"}
        );
        if (peek().kind != TokenKind::RBrace) {
            fail(
                peek(),
                "conditional block must contain exactly one supported instruction"
            );
        }
        consume();
        return circuit;
    }

    [[nodiscard]] Circuit parse_measurement(
        Circuit circuit,
        std::optional<ClassicalCondition> condition
    ) {
        if (peek(1U).kind == TokenKind::LBracket) {
            const std::size_t classical_bit = parse_clbit_ref();
            expect(TokenKind::Assign, "'=' in measurement assignment");
            expect_identifier("measure");
            const std::size_t qubit = parse_qubit_ref();
            expect(TokenKind::Semicolon, "';' after measurement");
            return circuit.measure(qubit, classical_bit, condition);
        }

        const Token& classical_register = expect(
            TokenKind::Identifier,
            "classical register name"
        );
        if (classical_register.text != clbit_register_) {
            fail(
                classical_register,
                "unknown classical register '" + classical_register.text + "'"
            );
        }
        if (condition.has_value()) {
            fail(
                classical_register,
                "conditional whole-register measurement is not supported by the QuPy circuit IR"
            );
        }
        expect(TokenKind::Assign, "'=' in measurement assignment");
        expect_identifier("measure");
        const Token& quantum_register = expect(
            TokenKind::Identifier,
            "quantum register name"
        );
        if (quantum_register.text != qubit_register_) {
            fail(
                quantum_register,
                "unknown quantum register '" + quantum_register.text + "'"
            );
        }
        expect(TokenKind::Semicolon, "';' after measurement");
        if (num_clbits_ != num_qubits_) {
            fail(
                classical_register,
                "whole-register measurement requires equal quantum and classical register sizes"
            );
        }
        for (std::size_t index = 0U; index < num_qubits_; ++index) {
            circuit = circuit.measure(index, index);
        }
        return circuit;
    }

    [[nodiscard]] Circuit parse_barrier(
        Circuit circuit,
        std::optional<ClassicalCondition> condition
    ) {
        if (condition.has_value()) {
            fail(peek(), "conditional barrier is not supported by the QuPy circuit IR");
        }
        std::vector<std::size_t> qubits;
        if (peek().kind != TokenKind::Semicolon) {
            qubits.push_back(parse_qubit_ref());
            while (peek().kind == TokenKind::Comma) {
                consume();
                qubits.push_back(parse_qubit_ref());
            }
        }
        expect(TokenKind::Semicolon, "';' after barrier");
        return circuit.barrier(qubits);
    }

    [[nodiscard]] Circuit parse_gate(
        Circuit circuit,
        const Token& operation,
        std::optional<ClassicalCondition> condition
    ) {
        const std::string& name = operation.text;
        if (name == "h" || name == "x" || name == "y" || name == "z" || name == "reset") {
            const std::size_t qubit = parse_qubit_ref();
            expect(TokenKind::Semicolon, "';' after " + name);
            if (name == "h") return circuit.h(qubit, condition);
            if (name == "x") return circuit.x(qubit, condition);
            if (name == "y") return circuit.y(qubit, condition);
            if (name == "z") return circuit.z(qubit, condition);
            return circuit.reset(qubit, condition);
        }
        if (name == "rx" || name == "ry" || name == "rz") {
            expect(TokenKind::LParen, "'(' after " + name);
            const Token& parameter = expect(TokenKind::Number, name + " parameter");
            const double angle = parse_double(parameter);
            expect(TokenKind::RParen, "')' after " + name + " parameter");
            const std::size_t qubit = parse_qubit_ref();
            expect(TokenKind::Semicolon, "';' after " + name);
            if (name == "rx") return circuit.rx(angle, qubit, condition);
            if (name == "ry") return circuit.ry(angle, qubit, condition);
            return circuit.rz(angle, qubit, condition);
        }
        if (name == "cx" || name == "cz" || name == "swap") {
            const std::size_t first = parse_qubit_ref();
            expect(TokenKind::Comma, "',' between " + name + " operands");
            const std::size_t second = parse_qubit_ref();
            expect(TokenKind::Semicolon, "';' after " + name);
            if (name == "cx") return circuit.cx(first, second, condition);
            if (name == "cz") return circuit.cz(first, second, condition);
            return circuit.swap(first, second, condition);
        }
        fail(operation, "unsupported OpenQASM instruction '" + name + "'");
    }

    [[nodiscard]] Circuit parse_statement(
        Circuit circuit,
        std::optional<ClassicalCondition> condition
    ) {
        if (peek().kind != TokenKind::Identifier) {
            fail(peek(), "expected a supported OpenQASM instruction");
        }
        if (peek().text == "if") {
            if (condition.has_value()) {
                fail(peek(), "nested classical control is not supported");
            }
            return parse_if(std::move(circuit));
        }
        if (!clbit_register_.empty() && peek().text == clbit_register_ &&
            (peek(1U).kind == TokenKind::LBracket ||
             peek(1U).kind == TokenKind::Assign)) {
            return parse_measurement(std::move(circuit), condition);
        }
        const Token operation = consume();
        if (operation.text == "barrier") {
            return parse_barrier(std::move(circuit), condition);
        }
        return parse_gate(std::move(circuit), operation, condition);
    }

    std::vector<Token> tokens_;
    std::size_t position_ = 0U;
    std::string qubit_register_;
    std::string clbit_register_;
    std::size_t num_qubits_ = 0U;
    std::size_t num_clbits_ = 0U;
};

}  // namespace

Circuit Circuit::from_openqasm3(const std::string& text) {
    return Parser(text).parse();
}

}  // namespace qupy
