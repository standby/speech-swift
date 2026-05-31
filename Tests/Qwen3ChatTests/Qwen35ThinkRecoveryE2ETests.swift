import XCTest
@testable import Qwen3Chat

/// Reproduces the on-device long-meeting summarization failure: the 0.8B model
/// would open a `<think>` block and not close it within the token budget, so
/// `generateStream` suppressed every token and returned an empty string — which
/// collapsed map-reduce condensation to 0 chars. After the fix, a generation
/// that produced tokens must yield non-empty visible text.
///
/// E2E: downloads the Qwen3.5-0.8B MLX model on first run.
final class E2EQwen35ThinkRecoveryTests: XCTestCase {

    private func loadOrSkip() async throws -> Qwen35MLXChat {
        try await Qwen35MLXChat.fromPretrained { p, s in print("[\(Int(p*100))%] \(s)") }
    }

    /// A condense-style prompt over a chunk of dense transcript — the exact
    /// shape that triggered empty output in the app's map-reduce path.
    func testCondenseChunkProducesNonEmptyOutput() async throws {
        let chat = try await loadOrSkip()

        // ~6k chars of diarized transcript, like one map-reduce chunk.
        let line = "[Speaker 1] We reviewed the Q3 pipeline and agreed to ship the data-sync fix by Friday. [Speaker 2] I'll own the migration; Janice takes QA. "
        let chunk = String(repeating: line, count: 40)

        let system = "You compress one part of a longer meeting transcript. Preserve every speaker name, decision, action item, owner, date, number, and commitment. Output tight factual bullet points. Do not add a preamble."
        let user = "Condense this transcript excerpt into concise factual bullets, keeping all names, decisions, action items, and numbers. Output only the bullets.\n\n\(chunk)"

        var out = ""
        let stream = chat.generateStream(
            messages: [ChatMessage(role: .system, content: system),
                       ChatMessage(role: .user, content: user)],
            sampling: ChatSamplingConfig(temperature: 0.3, topK: 20, maxTokens: 256))
        for try await delta in stream { out += delta }

        let trimmed = out.trimmingCharacters(in: .whitespacesAndNewlines)
        print("Condense output (\(trimmed.count) chars): \(trimmed.prefix(200))")
        XCTAssertFalse(trimmed.isEmpty,
            "generateStream must not return empty for a chunk that generates tokens (think-recovery)")
    }
}
