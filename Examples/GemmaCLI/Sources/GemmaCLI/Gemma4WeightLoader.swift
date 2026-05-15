import Metal
import MTLSafeTensors
import Popcorn

// MARK: - WeightLoader

/// Hides MLX-vs-Transformers tensor naming and per-tensor quantization detection.
///
/// MLX exports prefix language-model weights as `language_model.model.*`, while the
/// Hugging Face Transformers export wraps them as `model.language_model.*`. We detect
/// the prefix at load time. Within either layout, a tensor `<name>.weight` is
/// considered affine-quantized when both `<name>.scales` and `<name>.biases` are
/// present in the archive (and the config has a top-level `quantization` block).
struct WeightLoader {
    // MARK: Lifecycle

    init(archive: SafeTensors, quantization: Gemma4Config.Quantization?) throws {
        self.archive = archive
        self.quantization = quantization

        // Detect the language-model prefix. MLX vs HF Transformers differ here.
        let allNames = archive.names
        if allNames.contains(where: { $0.hasPrefix("language_model.model.") }) {
            languageModelPrefix = "language_model.model"
        } else if allNames.contains(where: { $0.hasPrefix("model.language_model.") }) {
            languageModelPrefix = "model.language_model"
        } else {
            throw GemmaError.message("Could not find Gemma language model weights in archive.")
        }
    }

    // MARK: Internal

    let archive: SafeTensors
    let quantization: Gemma4Config.Quantization?

    /// Returns the fully qualified tensor name for a relative key within the language model.
    func qualify(_ relative: String) -> String {
        "\(languageModelPrefix).\(relative)"
    }

    func layerPrefix(_ index: Int) -> String {
        qualify("layers.\(index)")
    }

    /// Loads a linear weight by its absolute base name (without the `.weight` suffix),
    /// automatically picking between dense and quantized layouts.
    func linear(_ name: String) throws -> LinearWeight {
        if let quantized = try maybeQuantized(name) {
            return .quantized(quantized)
        }
        return try .dense(archive.popcornTensor("\(name).weight"))
    }

    /// Loads an embedding-table weight by its absolute base name.
    func embedding(_ name: String) throws -> EmbeddingTable {
        if let quantized = try maybeQuantized(name) {
            return .quantized(quantized)
        }
        return try .dense(archive.popcornTensor("\(name).weight"))
    }

    // MARK: Private

    private let languageModelPrefix: String

    private static func aligned4ByteBuffer(_ source: any MTLBuffer) throws -> any MTLBuffer {
        let address = UInt(bitPattern: source.contents())
        if address.isMultiple(of: 4) {
            return source
        }
        guard let copy = source.device.makeBuffer(length: source.length, options: [.storageModeShared]) else {
            throw GemmaError.message("Failed to allocate aligned copy of quantized weights buffer (\(source.length) bytes).")
        }
        copy.label = source.label
        memcpy(copy.contents(), source.contents(), source.length)
        return copy
    }

    private func maybeQuantized(_ name: String) throws -> AffineQuantizedTensor? {
        guard let quantization else { return nil }
        let weightName = "\(name).weight"
        let scalesName = "\(name).scales"
        let biasesName = "\(name).biases"
        let allNames = archive.names
        guard
            allNames.contains(weightName),
            allNames.contains(scalesName),
            allNames.contains(biasesName)
        else { return nil }

        let packed = try archive.tensor(named: weightName)
        guard packed.dtype == .u32 else {
            throw GemmaError.message("Expected U32 packed weights for \(weightName), got \(packed.dtype).")
        }
        let scales = try archive.popcornTensor(scalesName)
        let biases = try archive.popcornTensor(biasesName)

        // Safetensors packs tensors contiguously with no padding, so a U32 tensor
        // following a BF16 tensor with an odd element count lands at a 2-byte but
        // not 4-byte file offset. mmap-backed MTLBuffer views inherit that
        // alignment, and 32-bit kernel reads silently round the address down.
        // Copy into a freshly allocated buffer when the underlying contents are
        // not 4-byte aligned.
        let weightsBuffer = try Self.aligned4ByteBuffer(packed.buffer)
        let packedValues = Tensor(buffer: weightsBuffer, shape: packed.shape, dataType: packed.dtype.popcornDataType)

        let perWord = 32 / quantization.bits
        let outFeatures = packed.shape[0]
        let packedInner = packed.shape[1]
        let inFeatures = packedInner * perWord
        return try AffineQuantizedTensor(
            packedValues: packedValues,
            scales: scales,
            biases: biases,
            shape: Tensor.Shape([outFeatures, inFeatures]),
            format: .init(bits: quantization.bits, groupSize: quantization.groupSize, packing: .uint32LittleEndian)
        )
    }
}

// MARK: - SafeTensors helpers

extension SafeTensors {
    func popcornTensor(_ name: String) throws -> Tensor {
        let safeTensor = try tensor(named: name)
        return Tensor(buffer: safeTensor.buffer, shape: safeTensor.shape, dataType: safeTensor.dtype.popcornDataType)
    }
}

extension SafeTensors.DType {
    var popcornDataType: Tensor.DataType {
        switch self {
        case .u8, .bool: .u8
        case .u16: .u16
        case .u32: .u32
        case .u64: .u64
        case .i8: .i8
        case .i16: .i16
        case .i32: .i32
        case .i64: .i64
        case .f16: .f16
        case .bf16: .bf16
        case .f32: .f32
        case .f64: .f32
        }
    }
}
