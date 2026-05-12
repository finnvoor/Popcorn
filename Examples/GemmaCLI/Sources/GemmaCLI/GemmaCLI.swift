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

    @Option(name: .customLong("backend"), help: "GPU backend: metal or metal4.") var backend: KernelBackend = .metal4

    @Option(
        name: .customLong("capture"),
        help: "Write a Metal GPU trace to this path (e.g. trace.gputrace). Programmatic capture requires the METAL_CAPTURE_ENABLED=1 environment variable."
    ) var capturePath: String?

    @Option(
        name: .customLong("profile"),
        help: "Write per-dispatch GPU timings to this JSONL file (Metal 4 backend only)."
    ) var profilePath: String?

    func run() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw GemmaError.message("No Metal device found.")
        }
        if backend == .metal4, !device.supportsFamily(.metal4) {
            throw GemmaError.message("\(device.name) does not support Metal 4.")
        }

        let captureSession = try beginCapture(device: device)
        defer { captureSession?.end() }

        let modelDirectory = try await resolveModelDirectory()
        FileHandle.standardError.write(Data("Loading model...\n".utf8))
        let loadStart = ContinuousClock.now
        let tokenizer = try await AutoTokenizer.from(modelFolder: modelDirectory, strict: false)
        let model = try Gemma4TextInference(
            device: device,
            modelDirectory: modelDirectory,
            maxSeqLen: maxSeqLen,
            backend: backend
        )
        let loadDuration = ContinuousClock.now - loadStart
        let loadSeconds = Double(loadDuration.components.seconds) + Double(loadDuration.components.attoseconds) / 1e18
        FileHandle.standardError.write(Data(String(format: "Model loaded in %.2fs\n", loadSeconds).utf8))

        if let profilePath {
            guard backend == .metal4, let metal4Context = model.context as? Metal4KernelContext else {
                throw GemmaError.message("--profile requires the metal4 backend.")
            }
            let outputURL = URL(fileURLWithPath: (profilePath as NSString).expandingTildeInPath)
            if FileManager.default.fileExists(atPath: outputURL.path) {
                try FileManager.default.removeItem(at: outputURL)
            }
            metal4Context.profiler = try Metal4DispatchProfiler(device: device, outputURL: outputURL)
            FileHandle.standardError.write(Data("Recording per-dispatch profile to \(outputURL.path)\n".utf8))
        }

        var tokenIds = tokenizer.encode(text: prompt, addSpecialTokens: true)
        guard !tokenIds.isEmpty else { throw GemmaError.message("Tokenizer produced no tokens.") }
        guard tokenIds.count < maxSeqLen else {
            throw GemmaError.message("Prompt has \(tokenIds.count) tokens, exceeding max sequence length \(maxSeqLen).")
        }

        FileHandle.standardOutput.write(Data(prompt.utf8))

        let promptTokens = tokenIds.count
        var generatedTokens = 0

        let start = ContinuousClock.now
        let prefill = try model.submitPrefill(tokenIds)
        let firstSampled = try prefill.wait()
        let promptSeconds = elapsedSeconds(since: start)
        let generationStart = ContinuousClock.now
        generatedTokens = 1

        let printSampled: (Int) -> Void = { token in
            let piece = tokenizer.decode(tokens: [token], skipSpecialTokens: true)
            FileHandle.standardOutput.write(Data(piece.utf8))
        }
        printSampled(firstSampled)
        tokenIds.append(firstSampled)

        // Pipeline `lookahead + 1` decode CBs in flight at once. Each decode CB reads its
        // input id from the previous CB's GPU-side output ring slot, so there's no
        // CPU-side data dependency and the GPU stays busy across token boundaries.
        // When profiling we force sequential execution so the GPU counter heap can
        // serve one CB at a time and timestamps aren't interleaved across CBs.
        let lookahead = profilePath == nil ? 2 : 0
        var pending: [Gemma4TextInference.PendingForward] = []
        var nextOffset = tokenIds.count

        func enqueueDecode() throws -> Bool {
            guard generatedTokens + pending.count < maxTokens, nextOffset < maxSeqLen else { return false }
            try pending.append(model.submitDecodeStep(at: nextOffset))
            nextOffset += 1
            return true
        }

        if firstSampled != model.config.eosTokenId {
            for _ in 0...lookahead {
                if try !enqueueDecode() { break }
            }
        }

        loop: while !pending.isEmpty {
            let next = pending.removeFirst()
            let sampled = try next.wait()
            generatedTokens += 1
            printSampled(sampled)
            tokenIds.append(sampled)

            if sampled == model.config.eosTokenId { break loop }

            _ = try enqueueDecode()
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

    private func beginCapture(device: any MTLDevice) throws -> CaptureSession? {
        guard let capturePath else { return nil }
        let manager = MTLCaptureManager.shared()
        guard manager.supportsDestination(.gpuTraceDocument) else {
            throw GemmaError.message("This system does not support writing GPU trace documents.")
        }

        let outputURL = URL(fileURLWithPath: (capturePath as NSString).expandingTildeInPath)
        if FileManager.default.fileExists(atPath: outputURL.path) {
            try FileManager.default.removeItem(at: outputURL)
        }

        let descriptor = MTLCaptureDescriptor()
        descriptor.captureObject = device
        descriptor.destination = .gpuTraceDocument
        descriptor.outputURL = outputURL
        do {
            try manager.startCapture(with: descriptor)
        } catch {
            throw GemmaError.message("Failed to start GPU capture. Run with METAL_CAPTURE_ENABLED=1 in the environment. (\(error.localizedDescription))")
        }
        FileHandle.standardError.write(Data("Capturing GPU trace to \(outputURL.path)\n".utf8))
        return CaptureSession(manager: manager, outputURL: outputURL)
    }

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
        FileHandle.standardError.write(Data(report.utf8))
    }
}

// MARK: - CaptureSession

private final class CaptureSession {
    // MARK: Lifecycle

    init(manager: MTLCaptureManager, outputURL: URL) {
        self.manager = manager
        self.outputURL = outputURL
    }

    // MARK: Internal

    func end() {
        guard !ended else { return }
        ended = true
        manager.stopCapture()
        FileHandle.standardError.write(Data("Wrote GPU trace to \(outputURL.path)\n".utf8))
    }

    // MARK: Private

    private let manager: MTLCaptureManager
    private let outputURL: URL
    private var ended = false
}

private func elapsedSeconds(since instant: ContinuousClock.Instant) -> Double {
    let elapsed = ContinuousClock.now - instant
    return Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
}
