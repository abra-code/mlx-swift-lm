// Copyright © 2026 Apple Inc.

import Testing

@testable import MLXLMCommon

/// The token loop's text path: reasoning is split out of the decoded stream into
/// ``Generation/reasoning(_:)`` before anything reaches the tool-call parser.
@Suite
struct GenerationReasoningRoutingTests {

    // MARK: - Fixtures & helpers

    private static let gemma4 = ReasoningConfig(
        startDelimiter: "<|channel>", endDelimiter: "<channel|>",
        promptStrategy: .templateFlag(key: "enable_thinking", defaultOn: false),
        isSpecialToken: true,
        channel: .gemma4)

    private static let think = ReasoningConfig(
        startDelimiter: "<think>", endDelimiter: "</think>", promptStrategy: .alwaysOn)

    /// Feeds decoded text through the handler and drains it, returning everything
    /// the stream would have yielded (minus the trailing `.info`).
    private func drive(
        reasoningConfig: ReasoningConfig?,
        format: ToolCallFormat = .json,
        primedInside: Bool = false,
        _ chunks: [String]
    ) -> [Generation] {
        var handler = TextToolTokenLoopHandler(
            tokenizer: TestTokenizer(),
            format: format,
            reasoningConfig: reasoningConfig,
            reasoningPrimedInside: primedInside)

        var events: [Generation] = []
        for chunk in chunks {
            _ = handler.processText(chunk) { event in
                events.append(event)
                return .enqueued(remaining: 1)
            }
        }
        handler.onGenerationEnd { event in
            events.append(event)
            return .enqueued(remaining: 1)
        }
        return events
    }

    private func chunkText(_ events: [Generation]) -> String {
        events.compactMap(\.chunk).joined()
    }

    private func reasoningText(_ events: [Generation]) -> String {
        events.compactMap(\.reasoning).joined()
    }

    private func toolCalls(_ events: [Generation]) -> [ToolCall] {
        events.compactMap(\.toolCall)
    }

    // MARK: - Gemma 4

    /// The production case: after a tool call the 31B template suppresses the whole
    /// generation prompt, prefill included, so the model is the one to open the
    /// channel and the raw markers land in `.chunk`.
    @Test func gemma4EmptyThoughtBlockDoesNotLeakIntoChunks() {
        let events = drive(reasoningConfig: Self.gemma4, ["<|channel>thought\n<channel|>Sunny."])
        #expect(chunkText(events) == "Sunny.")
        #expect(reasoningText(events).isEmpty)
    }

    @Test func gemma4ThoughtRoutesToReasoning() {
        let events = drive(
            reasoningConfig: Self.gemma4, ["<|channel>thought\nchecking<channel|>Sunny."])
        #expect(reasoningText(events) == "checking")
        #expect(chunkText(events) == "Sunny.")
    }

    /// A `responseLabels` channel routes to `.chunk`, not `.reasoning` (defensive:
    /// no shipped Gemma 4 template emits one).
    @Test func responseLabelledChannelIsAnAnswer() {
        let events = drive(reasoningConfig: Self.gemma4, ["<|channel>content\nSunny.<channel|>"])
        #expect(chunkText(events) == "Sunny.")
        #expect(reasoningText(events).isEmpty)
    }

    // MARK: - Ordering against the tool-call parser

    /// Reasoning must be split out FIRST. A model that discusses tool syntax in its
    /// scratchpad would otherwise produce a phantom tool call.
    @Test func toolSyntaxInsideReasoningIsNotAToolCall() {
        let events = drive(
            reasoningConfig: Self.think,
            [
                #"<think>I could call <tool_call>{"name":"rm","arguments":{}}</tool_call></think>No."#
            ])
        #expect(toolCalls(events).isEmpty)
        #expect(chunkText(events) == "No.")
        #expect(reasoningText(events).contains("rm"))
    }

    /// A real tool call after the thought block still parses.
    @Test func toolCallAfterReasoningStillParses() {
        let events = drive(
            reasoningConfig: Self.gemma4,
            [
                #"<|channel>thought\#nlooking it up<channel|><tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#
            ])
        #expect(reasoningText(events) == "looking it up")
        #expect(toolCalls(events).map(\.function.name) == ["get_weather"])
        #expect(chunkText(events).isEmpty)
    }

    // MARK: - Models with no reasoning protocol are untouched

    @Test func withoutReasoningConfigTextPassesThroughVerbatim() {
        let events = drive(reasoningConfig: nil, ["<|channel>thought\n<channel|>Sunny."])
        #expect(chunkText(events) == "<|channel>thought\n<channel|>Sunny.")
        #expect(reasoningText(events).isEmpty)
    }

    @Test func withoutReasoningConfigToolCallsStillParse() {
        let events = drive(
            reasoningConfig: nil, [#"pre<tool_call>{"name":"f","arguments":{}}</tool_call>post"#])
        #expect(toolCalls(events).map(\.function.name) == ["f"])
        #expect(chunkText(events) == "prepost")
    }

    // MARK: - Streaming edges

    @Test func delimitersSplitAcrossChunksDoNotLeak() {
        let events = drive(
            reasoningConfig: Self.gemma4, ["<|chan", "nel>thought\nhmm<chan", "nel|>Done."])
        #expect(reasoningText(events) == "hmm")
        #expect(chunkText(events) == "Done.")
    }

    /// An end delimiter that never arrives as text must not strand the buffer:
    /// `onGenerationEnd` drains the reasoning scanner.
    @Test func generationEndFlushesHeldBackReasoning() {
        let events = drive(reasoningConfig: Self.think, ["<think>cut off mid-thou"])
        #expect(reasoningText(events) == "cut off mid-thou")
        #expect(chunkText(events).isEmpty)
    }

    @Test func generationEndFlushesHeldBackResponse() {
        let events = drive(reasoningConfig: Self.think, ["answer ending in <thin"])
        #expect(chunkText(events) == "answer ending in <thin")
    }

    /// DeepSeek-R1 prefills `<think>` into the prompt, so the stream only ever
    /// carries the closing delimiter.
    @Test func primedPromptRoutesTheOpeningThought() {
        let events = drive(
            reasoningConfig: Self.think, primedInside: true, ["deliberating</think>The answer."])
        #expect(reasoningText(events) == "deliberating")
        #expect(chunkText(events) == "The answer.")
    }

    // MARK: - Case accessors

    @Test func reasoningIsNotAChunk() {
        let reasoning = Generation.reasoning("thinking")
        #expect(reasoning.chunk == nil)
        #expect(reasoning.reasoning == "thinking")
        #expect(reasoning.toolCall == nil)
        #expect(reasoning.info == nil)
        #expect(Generation.chunk("answer").reasoning == nil)
    }
}
