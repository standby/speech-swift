import XCTest
@testable import Qwen3Chat

/// Regression test for chunked prompt prefill (Qwen35MLXModel.prefill).
///
/// A single-pass prefill of a long prompt is one GPU command buffer long
/// enough to trip Metal's watchdog, which kills the app — e.g. when
/// summarizing a long meeting transcript. The fix processes the prompt in
/// fixed-size windows. That must be *numerically equivalent* to one pass:
/// with greedy decoding the generated token sequence must be identical
/// regardless of chunk size.
///
/// E2E: downloads the Qwen3.5-0.8B MLX model on first run.
final class E2EQwen35ChunkedPrefillTests: XCTestCase {

    func testChunkedPrefillMatchesSinglePass() async throws {
        let chat = try await Qwen35MLXChat.fromPretrained { progress, status in
            print("[\(Int(progress * 100))%] \(status)")
        }

        // A prompt longer than one chunk so the chunked path runs many
        // windows while the single-pass path is one big forward — the
        // exact configuration that used to trip the GPU watchdog.
        let longContext = String(
            repeating: "The quick brown fox jumps over the lazy dog. ", count: 80)
        let messages = [
            ChatMessage(role: .user,
                        content: longContext + "\nReply with the single word: ok.")
        ]
        let promptIds = ChatTemplate.encode(
            messages: messages, tokenizer: chat.tokenizer,
            config: chat.config, enableThinking: false)
        XCTAssertGreaterThan(
            promptIds.count, Qwen35MLXModel.prefillChunkSize,
            "Prompt must exceed one chunk to actually exercise chunking")

        // Greedy → deterministic, so equality is a meaningful assertion.
        let greedy = ChatSamplingConfig(temperature: 0, topK: 1, maxTokens: 24)

        let chunked = chat.model.generate(
            promptIds: promptIds, sampling: greedy, prefillChunkSize: 8)
        let singlePass = chat.model.generate(
            promptIds: promptIds, sampling: greedy,
            prefillChunkSize: promptIds.count + 1)

        XCTAssertFalse(chunked.isEmpty, "Generation should produce tokens")
        XCTAssertEqual(
            chunked, singlePass,
            "Chunked prefill must produce identical greedy output to single-pass")
        print("Chunked == single-pass over \(promptIds.count) prompt tokens; "
            + "generated \(chunked.count) tokens")
    }
}
