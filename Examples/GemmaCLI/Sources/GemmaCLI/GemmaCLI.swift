import ArgumentParser
import Foundation
import HuggingFace
import Metal
import Tokenizers

// MARK: - GemmaCLI

@main struct GemmaCLI: AsyncParsableCommand {
    // MARK: Internal

    static let configuration = CommandConfiguration(
        commandName: "gemma-cli",
        abstract: "Gemma 4 text inference using Popcorn Metal 4 kernels."
    )

    @Option(help: "Prompt text.") var prompt = "The capital of Ireland is"

    @Option(name: [.customLong("max-tokens"), .customLong("max-new")], help: "Max tokens to generate.") var maxTokens = 32

    @Option(name: .customLong("max-seq-len"), help: "Max KV-cache size.") var maxSeqLen = 512

    @Option(name: .customLong("model"), help: "Hugging Face repo id.") var modelId = "google/gemma-4-E2B"

    func run() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw GemmaError.message("No Metal device found.")
        }
        guard device.supportsFamily(.metal4) else {
            throw GemmaError.message("\(device.name) does not support Metal 4.")
        }

        let modelDirectory = try await resolveModelDirectory()
        let tokenizer = try await AutoTokenizer.from(modelFolder: modelDirectory, strict: false)
        let model = try Gemma4TextInference(
            device: device,
            modelDirectory: modelDirectory,
            maxSeqLen: maxSeqLen
        )

        var tokenIds = tokenizer.encode(text: prompt, addSpecialTokens: true)
        guard !tokenIds.isEmpty else { throw GemmaError.message("Tokenizer produced no tokens.") }
        guard tokenIds.count < maxSeqLen else {
            throw GemmaError.message("Prompt has \(tokenIds.count) tokens, exceeding max sequence length \(maxSeqLen).")
        }

        FileHandle.standardOutput.write(Data(prompt.utf8))

        let promptTokens = tokenIds.count
        var generatedTokens = 0

        let start = ContinuousClock.now
        let prefill = try model.submit(inputIds: tokenIds, offset: 0)
        var sampled = try prefill.wait()
        let promptSeconds = elapsedSeconds(since: start)
        let generationStart = ContinuousClock.now
        generatedTokens = 1

        loop: while true {
            if sampled == model.config.eosTokenId { break }
            tokenIds.append(sampled)

            let reachedTokenLimit = generatedTokens >= maxTokens
            let reachedSeqLimit = tokenIds.count >= maxSeqLen
            let pending: Gemma4TextInference.PendingForward? = (reachedTokenLimit || reachedSeqLimit)
                ? nil
                : try model.submit(inputIds: [sampled], offset: tokenIds.count - 1)

            let piece = tokenizer.decode(tokens: [sampled], skipSpecialTokens: true)
            FileHandle.standardOutput.write(Data(piece.utf8))

            guard let pending else { break loop }
            sampled = try pending.wait()
            generatedTokens += 1
        }
        let generationSeconds = elapsedSeconds(since: generationStart)

        FileHandle.standardOutput.write(Data("\n".utf8))
        printStatistics(
            promptTokens: promptTokens,
            promptSeconds: promptSeconds,
            generatedTokens: generatedTokens,
            generationSeconds: generationSeconds
        )
    }

    // MARK: Private

    private func resolveModelDirectory() async throws -> URL {
        guard let repo = Repo.ID(rawValue: modelId) else {
            throw GemmaError.message("Invalid Hugging Face repo id: \(modelId).")
        }
        return try await HubClient.default.downloadSnapshot(
            of: repo,
            matching: ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"],
            maxConcurrentDownloads: 4
        )
    }

    private func printStatistics(
        promptTokens: Int,
        promptSeconds: Double,
        generatedTokens: Int,
        generationSeconds: Double
    ) {
        let promptRate = promptSeconds > 0 ? Double(promptTokens) / promptSeconds : 0

        let generationRate = generationSeconds > 0 ? Double(generatedTokens) / generationSeconds : 0

        var report = "\n"
        report += String(format: "Prompt:     %4d tokens, %7.2f tok/s\n", promptTokens, promptRate)
        report += String(format: "Generation: %4d tokens, %7.2f tok/s\n", generatedTokens, generationRate)
        let calls = Gemma4TextInference.debugCallCount
        if calls > 0 {
            let encMs = Gemma4TextInference.debugEncodeSeconds * 1000 / Double(calls)
            let waitMs = Gemma4TextInference.debugCommitWaitSeconds * 1000 / Double(calls)
            let gpuMs = Gemma4TextInference.debugGPUSeconds * 1000 / Double(calls)
            report += String(format: "Per submit: encode %.2f ms, wait %.2f ms, gpu %.2f ms (%d calls)\n", encMs, waitMs, gpuMs, calls)
        }
        FileHandle.standardError.write(Data(report.utf8))
    }
}

private func elapsedSeconds(since instant: ContinuousClock.Instant) -> Double {
    let elapsed = ContinuousClock.now - instant
    return Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
}
