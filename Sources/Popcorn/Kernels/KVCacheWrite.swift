import Metal
import PopcornShaderTypes

public extension Kernels {
    struct KVCacheWrite: Kernel {
        // MARK: Lifecycle

        public init(
            source: Tensor,
            cache: Tensor,
            batch: Int,
            kvHeads: Int,
            newLen: Int,
            maxLen: Int,
            headDim: Int,
            offset: Int,
        ) throws {
            self.source = source
            self.cache = cache
            functionName = switch (source.dataType, cache.dataType) {
            case (.f32, .f32): "kv_cache_write"
            case (.bf16, .bf16): "kv_cache_write_bf16"
            case (.f32, .bf16): "kv_cache_write_f32_to_bf16"
            default: throw PopcornError.unsupportedDataTypeCombination("Unsupported KV cache data type combination: \(source.dataType), \(cache.dataType).")
            }
            constants = [KVCacheWriteConstants(
                B: UInt32(batch),
                Nkv: UInt32(kvHeads),
                Snew: UInt32(newLen),
                Smax: UInt32(maxLen),
                Hd: UInt32(headDim),
                offset: UInt32(offset)
            )]
            grid = MTLSize(width: batch * kvHeads * newLen * headDim, height: 1, depth: 1)
        }

        public init(source: Tensor, cache: Tensor, offset: Int) throws {
            guard source.shape.rank == 4 else { throw PopcornError.tensorInvalidRank(expected: 4, actual: source.shape.rank) }
            guard cache.shape.rank == 4 else { throw PopcornError.tensorInvalidRank(expected: 4, actual: cache.shape.rank) }
            let batch = source.shape[0], kvHeads = source.shape[1], newLen = source.shape[2], headDim = source.shape[3]
            guard cache.shape[0] == batch, cache.shape[1] == kvHeads, cache.shape[3] == headDim else {
                throw PopcornError.tensorShapeMismatch("KV cache shape must match source batch/heads/headDim; source \(source.shape.dimensions), cache \(cache.shape.dimensions).")
            }
            guard offset >= 0, offset + newLen <= cache.shape[2] else {
                throw PopcornError.tensorShapeMismatch("KV cache write range \(offset)..<\(offset + newLen) exceeds cache length \(cache.shape[2]).")
            }
            try self.init(source: source, cache: cache, batch: batch, kvHeads: kvHeads, newLen: newLen, maxLen: cache.shape[2], headDim: headDim, offset: offset)
        }

        // MARK: Public

        public let functionName: String
        public let constants: [any BitwiseCopyable]
        public let grid: MTLSize
        public let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)

        public var tensors: [Tensor.Binding] {
            [
                .init(tensor: source, access: .read),
                .init(tensor: cache, access: .write)
            ]
        }

        // MARK: Private

        private let source: Tensor
        private let cache: Tensor
    }
}
