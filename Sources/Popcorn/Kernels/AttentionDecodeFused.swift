import Foundation
import Metal
import PopcornShaderTypes

public extension Kernels {
    /// Single-pass fused attention for the decode path (Q seq len = 1).
    /// Combines `AttentionScoresSoftmax` + `AttentionOutput` into one
    /// streaming kernel — no intermediate `probs` materialization, one read of
    /// each K/V cache row, one barrier per output element.
    ///
    /// `q` shape: `[B, Nq, 1, Hd]`. `k`/`v` shape: `[B, Nkv, Sk, Hd]`. Output
    /// shape: `[B, Nq, 1, Hd]`. Only `Hd ∈ {256, 512}` is currently shipped.
    struct AttentionDecodeFused: Kernel {
        // MARK: Lifecycle

        public init(
            q: Tensor,
            k: Tensor,
            v: Tensor,
            out: Tensor,
            scale: Float,
            slidingWindow: Int? = nil
        ) throws {
            guard q.shape.rank == 4, k.shape.rank == 4, v.shape.rank == 4 else {
                throw PopcornError.tensorInvalidRank(expected: 4, actual: q.shape.rank)
            }
            let b = q.shape[0], nq = q.shape[1], sq = q.shape[2], hd = q.shape[3]
            let nkv = k.shape[1], sk = k.shape[2]
            guard sq == 1 else {
                throw PopcornError.tensorShapeMismatch("AttentionDecodeFused requires Sq=1; got \(sq).")
            }
            guard k.shape[0] == b, k.shape[3] == hd, v.shape[0] == b, v.shape[1] == nkv, v.shape[2] == sk, v.shape[3] == hd else {
                throw PopcornError.tensorShapeMismatch("k/v shape mismatch.")
            }
            guard nq.isMultiple(of: nkv) else {
                throw PopcornError.tensorShapeMismatch("Nq must be a multiple of Nkv.")
            }
            guard out.shape.dimensions == [b, nq, 1, hd] else {
                throw PopcornError.tensorShapeMismatch("out shape must be [\(b), \(nq), 1, \(hd)]; got \(out.shape.dimensions).")
            }

            self.q = q
            self.k = k
            self.v = v
            self.out = out

            functionName = try Self.functionName(qDataType: q.dataType, kvDataType: k.dataType, hd: hd)
            constants = [AttentionDecodeFusedConstants(
                B: UInt32(b),
                Nq: UInt32(nq),
                Nkv: UInt32(nkv),
                Sk: UInt32(sk),
                Hd: UInt32(hd),
                slidingWindow: Int32(slidingWindow ?? -1),
                scale: scale
            )]
            // One TG of 1024 threads (32 simdgroups × 32 lanes) per (B, Nq).
            grid = MTLSize(width: 1024, height: b * nq, depth: 1)
            threadgroupSize = MTLSize(width: 1024, height: 1, depth: 1)
        }

        public static func supports(qDataType: Tensor.DataType, kvDataType: Tensor.DataType, hd: Int) -> Bool {
            (try? functionName(qDataType: qDataType, kvDataType: kvDataType, hd: hd)) != nil
        }

        // MARK: Public

        public let functionName: String
        public let constants: [any BitwiseCopyable]
        public let grid: MTLSize
        public let threadgroupSize: MTLSize

        public var tensors: [Tensor.Binding] {
            [
                .init(tensor: q, access: .read),
                .init(tensor: k, access: .read),
                .init(tensor: v, access: .read),
                .init(tensor: out, access: .write)
            ]
        }

        // MARK: Private

        private let q: Tensor
        private let k: Tensor
        private let v: Tensor
        private let out: Tensor

        private static func functionName(qDataType: Tensor.DataType, kvDataType: Tensor.DataType, hd: Int) throws -> String {
            switch (qDataType, kvDataType, hd) {
            case (.bf16, .bf16, 256): "attention_decode_fused_bf16_D256"
            case (.bf16, .bf16, 512): "attention_decode_fused_bf16_D512"
            default:
                throw PopcornError.unsupportedDataTypeCombination("Unsupported AttentionDecodeFused: q=\(qDataType), kv=\(kvDataType), Hd=\(hd).")
            }
        }
    }
}
