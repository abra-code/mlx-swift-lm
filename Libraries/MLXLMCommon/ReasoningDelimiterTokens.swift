// Copyright © 2026 Apple Inc.

/// A reasoning protocol's two framing delimiters as single token ids.
///
/// Matching the ids instead of their text keeps a delimiter the model only
/// spells out, for example when it explains its own format, as ordinary text.
struct ReasoningDelimiterTokens: Sendable {
    let start: Int
    let end: Int
    private let startDelimiter: String
    private let endDelimiter: String

    /// `nil` unless both delimiters are single tokens that decode back to themselves.
    /// The caller then falls back to scanning the decoded text.
    init?(config: ReasoningConfig, tokenizer: any Tokenizer) {
        guard let start = Self.id(of: config.startDelimiter, tokenizer: tokenizer),
            let end = Self.id(of: config.endDelimiter, tokenizer: tokenizer),
            start != end
        else { return nil }
        self.start = start
        self.end = end
        self.startDelimiter = config.startDelimiter
        self.endDelimiter = config.endDelimiter
    }

    func text(of boundary: ReasoningEventEmitter.Boundary) -> String {
        boundary == .start ? startDelimiter : endDelimiter
    }

    func boundary(for token: Int) -> ReasoningEventEmitter.Boundary? {
        if token == start { return .start }
        if token == end { return .end }
        return nil
    }

    private static func id(of delimiter: String, tokenizer: any Tokenizer) -> Int? {
        guard let id = tokenizer.singleTokenID(for: delimiter),
            tokenizer.decode(tokenIds: [id]) == delimiter
        else { return nil }
        return id
    }
}
