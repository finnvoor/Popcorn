import Metal

// MARK: - AffineQuantizedTensor

/// A tensor stored using block-affine integer quantization: `w = q * scale + bias`,
/// where `q` is a small unsigned integer (`bits` wide) and a `scale`/`bias` pair is
/// shared across each contiguous group of `groupSize` values along the inner
/// (input-feature) axis.
///
/// Represents block-affine formats where values can be reconstructed as
/// `q * scale + bias`. Current kernels support MLX-style unsigned 4-bit values
/// packed into `u32` words with the lowest-index value occupying the least
/// significant bits, packed along the inner axis.
///
/// `packedValues` holds the packed quantized stream; `scales` and `biases` are
/// dense floating-point tensors of shape `[..., outFeatures, inFeatures / groupSize]`.
/// `shape` is the logical (dequantized) shape, with the last two dimensions being
/// `[outFeatures, inFeatures]`.
public struct AffineQuantizedTensor {
    // MARK: Lifecycle

    public init(
        packedValues: Tensor,
        scales: Tensor,
        biases: Tensor?,
        shape: Tensor.Shape,
        format: AffineQuantizationFormat
    ) throws {
        guard shape.rank >= 2 else {
            throw PopcornError.tensorInvalidRank(expected: 2, actual: shape.rank)
        }
        if let biases {
            guard biases.dataType == scales.dataType else {
                throw PopcornError.unsupportedDataTypeCombination(
                    "AffineQuantizedTensor scales and biases must share a data type; got \(scales.dataType) and \(biases.dataType)."
                )
            }
            guard biases.shape == scales.shape else {
                throw PopcornError.tensorShapeMismatch(
                    "AffineQuantizedTensor scales and biases must share a shape; got \(scales.shape.dimensions) and \(biases.shape.dimensions)."
                )
            }
        }
        guard packedValues.dataType == format.packing.storageDataType else {
            throw PopcornError.unsupportedDataTypeCombination(
                "AffineQuantizedTensor packed values must use \(format.packing.storageDataType) for packing \(format.packing); got \(packedValues.dataType)."
            )
        }
        self.packedValues = packedValues
        self.scales = scales
        self.biases = biases
        self.shape = shape
        self.format = format
    }

    // MARK: Public

    /// Packed quantized values. For `.uint32LittleEndian` 4-bit affine quantization,
    /// this is a `u32` tensor with shape `[..., outFeatures, inFeatures / 8]`.
    public let packedValues: Tensor

    /// Per-group scales, shape `[..., outFeatures, inFeatures / groupSize]`.
    public let scales: Tensor

    /// Per-group zero/bias offsets, shape `[..., outFeatures, inFeatures / groupSize]`.
    /// `nil` for symmetric formats; currently the bundled kernels expect a non-nil value.
    public let biases: Tensor?

    /// Logical (dequantized) shape, last two dimensions are `[outFeatures, inFeatures]`.
    public let shape: Tensor.Shape

    /// Quantization format parameters.
    public let format: AffineQuantizationFormat

    public var outFeatures: Int { shape[shape.rank - 2] }
    public var inFeatures: Int { shape[shape.rank - 1] }

    /// All Metal buffers backing this tensor. Useful for residency setup.
    public var allBuffers: [any MTLBuffer] {
        var buffers: [any MTLBuffer] = [packedValues.buffer, scales.buffer]
        if let biases { buffers.append(biases.buffer) }
        return buffers
    }
}
