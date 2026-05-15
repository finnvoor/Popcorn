import Metal
import Popcorn

// MARK: - EmbeddingTable

/// A lookup table that is either a dense `Tensor` (`[vocab, hidden]`) or an
/// `AffineQuantizedTensor` with the same logical shape.
enum EmbeddingTable {
    case dense(Tensor)
    case quantized(AffineQuantizedTensor)

    // MARK: Internal

    var hiddenSize: Int {
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

    /// Encodes `out[i, :] = table[ids[i], :]` (dequantizing on the fly when needed).
    func encodeGather(ids: Tensor, into out: Tensor, on encoder: KernelCommandEncoder) throws {
        switch self {
        case let .dense(t):
            try encoder.encode(Kernels.EmbeddingGather(ids: ids, table: t, into: out))
        case let .quantized(q):
            try encoder.encode(Kernels.EmbeddingGather(ids: ids, table: q, into: out))
        }
    }

    /// Encodes `out = x @ table^T`, i.e. an LM head over the tied embedding table.
    func encodeLMHead(_ x: Tensor, into out: Tensor, on encoder: KernelCommandEncoder) throws {
        switch self {
        case let .dense(t):
            try encoder.encode(Kernels.Matmul(x, t, into: out, transposeB: true))
        case let .quantized(q):
            try encoder.encode(Kernels.Matmul(x, q, into: out))
        }
    }
}
