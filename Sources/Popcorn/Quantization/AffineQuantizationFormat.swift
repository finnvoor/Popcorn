// MARK: - AffineQuantizationFormat

/// Describes a block-affine quantized storage format.
///
/// The current kernels support the MLX-style subset: unsigned 4-bit values packed
/// little-endian into `u32` words, with one scale/bias pair per contiguous group
/// along the inner axis.
public struct AffineQuantizationFormat: Sendable, Equatable {
    // MARK: Lifecycle

    public init(
        bits: Int,
        groupSize: Int,
        packing: Packing = .uint32LittleEndian
    ) {
        self.bits = bits
        self.groupSize = groupSize
        self.packing = packing
    }

    // MARK: Public

    /// How quantized integer values are packed into `packedValues`.
    public enum Packing: Sendable, Equatable {
        /// Unsigned values packed least-significant bits first into little-endian `u32` words.
        ///
        /// For example, with `bits == 4`, logical value `i` occupies bits
        /// `(i % 8) * 4 ..< (i % 8 + 1) * 4` of `packedValues[i / 8]`.
        case uint32LittleEndian
    }

    /// Bits per quantized value. Current kernels support `4`.
    public var bits: Int

    /// Number of consecutive inner-axis values sharing a scale/bias pair.
    public var groupSize: Int

    /// Packed storage convention for the quantized integer stream.
    public var packing: Packing

    /// Number of quantized values held by one packed storage element.
    public var valuesPerPackedElement: Int {
        packing.storageBitWidth / bits
    }
}

public extension AffineQuantizationFormat.Packing {
    var storageDataType: Tensor.DataType {
        switch self {
        case .uint32LittleEndian: .u32
        }
    }

    var storageBitWidth: Int {
        switch self {
        case .uint32LittleEndian: 32
        }
    }
}
