import Metal
import Popcorn

// MARK: - LinearWeight

/// A linear-layer weight that is either a dense `Tensor` (transposed at matmul time)
/// or an `AffineQuantizedTensor`. Both shapes follow `[outFeatures, inFeatures]`.
enum LinearWeight {
    case dense(Tensor)
    case quantized(AffineQuantizedTensor)

    // MARK: Internal

    var outFeatures: Int {
        switch self {
        case let .dense(t): t.shape[0]
        case let .quantized(q): q.outFeatures
        }
    }

    var inFeatures: Int {
        switch self {
        case let .dense(t): t.shape[1]
        case let .quantized(q): q.inFeatures
        }
    }

    var allBuffers: [MTLBuffer] {
        switch self {
        case let .dense(t): [t.buffer]
        case let .quantized(q): q.allBuffers
        }
    }

    /// Encodes `out = x @ weight^T` using the appropriate Popcorn kernel.
    func encodeMatmul(_ x: Tensor, into out: Tensor, on encoder: KernelCommandEncoder) throws {
        switch self {
        case let .dense(t):
            try encoder.encode(Kernels.Matmul(x, t, into: out, transposeB: true))
        case let .quantized(q):
            try encoder.encode(Kernels.Matmul(x, q, into: out))
        }
    }
}
