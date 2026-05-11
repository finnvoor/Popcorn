import Foundation
import Metal
import PopcornShaderTypes

public extension Kernels {
    /// Fused SwiGLU matvec for the t=1 decode path of a SwiGLU MLP:
    ///   `out[n] = gelu_tanh(dot(x, gate[n, :])) * dot(x, up[n, :])`
    /// Replaces the gate matvec + up matvec + gelu + mul sequence with one
    /// dispatch and reads x only once across both projections.
    struct SwigluMatvec: Kernel {
        // MARK: Lifecycle

        public init(x: Tensor, gate: Tensor, up: Tensor, out: Tensor) throws {
            guard x.shape.rank == 2, x.shape[0] == 1 else {
                throw PopcornError.tensorShapeMismatch("SwigluMatvec requires x of shape [1, K]; got \(x.shape.dimensions).")
            }
            guard gate.shape.rank == 2, up.shape.rank == 2 else {
                throw PopcornError.tensorInvalidRank(expected: 2, actual: gate.shape.rank)
            }
            let k = x.shape[1]
            let n = gate.shape[0]
            guard gate.shape.dimensions == [n, k], up.shape.dimensions == [n, k] else {
                throw PopcornError.tensorShapeMismatch("SwigluMatvec gate/up must both be [N, K] with matching shapes; got \(gate.shape.dimensions) / \(up.shape.dimensions).")
            }
            guard out.shape.elementCount == n else {
                throw PopcornError.tensorShapeMismatch("SwigluMatvec output must have N elements; got \(out.shape.dimensions).")
            }
            guard gate.dataType == up.dataType else {
                throw PopcornError.unsupportedDataTypeCombination("SwigluMatvec gate/up dtypes must match.")
            }

            self.x = x
            self.gate = gate
            self.up = up
            self.out = out

            functionName = try Self.functionName(x: x.dataType, w: gate.dataType, o: out.dataType)
            constants = [MatvecConstants(K: UInt32(k), N: UInt32(n), transposeW: 1)]
            let rowsPerSimdgroup = 4
            grid = MTLSize(width: ((n + rowsPerSimdgroup - 1) / rowsPerSimdgroup) * 32, height: 1, depth: 1)
            threadgroupSize = MTLSize(width: 128, height: 1, depth: 1)
        }

        public static func supports(x: Tensor.DataType, w: Tensor.DataType, o: Tensor.DataType) -> Bool {
            (try? functionName(x: x, w: w, o: o)) != nil
        }

        // MARK: Public

        public let functionName: String
        public let constants: [any BitwiseCopyable]
        public let grid: MTLSize
        public let threadgroupSize: MTLSize

        public var tensors: [Tensor.Binding] {
            [
                .init(tensor: x, access: .read),
                .init(tensor: gate, access: .read),
                .init(tensor: up, access: .read),
                .init(tensor: out, access: .write)
            ]
        }

        // MARK: Private

        private let x: Tensor
        private let gate: Tensor
        private let up: Tensor
        private let out: Tensor

        private static func functionName(x: Tensor.DataType, w: Tensor.DataType, o: Tensor.DataType) throws -> String {
            switch (x, w, o) {
            case (.bf16, .bf16, .bf16): "swiglu_matvec_bf16_bf16_bf16"
            case (.bf16, .bf16, .f32):  "swiglu_matvec_bf16_bf16_f32"
            case (.f32,  .bf16, .f32):  "swiglu_matvec_f32_bf16_f32"
            default:
                throw PopcornError.unsupportedDataTypeCombination("Unsupported SwigluMatvec dtypes: x=\(x), w=\(w), o=\(o).")
            }
        }
    }
}
