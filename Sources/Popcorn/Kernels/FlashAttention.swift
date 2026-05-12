import Metal
import PopcornShaderTypes

// MARK: - Kernels.FlashAttention

public extension Kernels {
    /// FlashAttention router: fused attention for prefill and split-KV decoding
    /// for single-token decode. MPP implementation details and scratch are
    /// managed internally.
    struct FlashAttention: Kernel {
        // MARK: Lifecycle

        public init(
            q: Tensor,
            k: Tensor,
            v: Tensor,
            out: Tensor,
            batch: Int,
            queryHeads: Int,
            kvHeads: Int,
            queryLen: Int,
            keyLen: Int,
            headDim: Int,
            scale: Float,
            slidingWindow: Int? = nil
        ) throws {
            guard queryHeads.isMultiple(of: kvHeads) else {
                throw PopcornError.tensorShapeMismatch(
                    "FlashAttention query heads must be divisible by KV heads; got \(queryHeads) and \(kvHeads)."
                )
            }
            guard headDim <= 512 else {
                throw PopcornError.tensorShapeMismatch(
                    "FlashAttention head dim must be <= 512 (kernel limit); got \(headDim)."
                )
            }
            self.q = q
            self.k = k
            self.v = v
            self.out = out
            self.batch = batch
            self.queryHeads = queryHeads
            self.kvHeads = kvHeads
            self.queryLen = queryLen
            self.keyLen = keyLen
            self.headDim = headDim
            self.scale = scale
            self.slidingWindow = slidingWindow
        }

        public init(
            q: Tensor,
            k: Tensor,
            v: Tensor,
            into out: Tensor,
            scale: Float,
            slidingWindow: Int? = nil
        ) throws {
            guard q.shape.rank == 4 else { throw PopcornError.tensorInvalidRank(expected: 4, actual: q.shape.rank) }
            guard k.shape.rank == 4 else { throw PopcornError.tensorInvalidRank(expected: 4, actual: k.shape.rank) }
            guard v.shape.rank == 4 else { throw PopcornError.tensorInvalidRank(expected: 4, actual: v.shape.rank) }
            let batch = q.shape[0]
            let queryHeads = q.shape[1]
            let queryLen = q.shape[2]
            let headDim = q.shape[3]
            let kvHeads = k.shape[1]
            let keyLen = k.shape[2]
            guard k.shape[0] == batch, k.shape[3] == headDim else {
                throw PopcornError.tensorShapeMismatch(
                    "FlashAttention K shape must match Q batch/headDim; q \(q.shape.dimensions), k \(k.shape.dimensions)."
                )
            }
            guard v.shape == k.shape else {
                throw PopcornError.tensorShapeMismatch(
                    "FlashAttention V shape must match K; v \(v.shape.dimensions), k \(k.shape.dimensions)."
                )
            }
            let expectedOut = [batch, queryHeads, queryLen, headDim]
            guard out.shape.dimensions == expectedOut else {
                throw PopcornError.tensorShapeMismatch(
                    "FlashAttention output shape must be \(expectedOut); got \(out.shape.dimensions)."
                )
            }
            try self.init(
                q: q, k: k, v: v, out: out,
                batch: batch, queryHeads: queryHeads, kvHeads: kvHeads,
                queryLen: queryLen, keyLen: keyLen, headDim: headDim,
                scale: scale, slidingWindow: slidingWindow
            )
        }

        // MARK: Public

        public enum Path: Equatable {
            case plain
            case mpp
            case decoding
        }

        public var tensors: [Tensor.Binding] {
            [
                .init(tensor: q, access: .read),
                .init(tensor: k, access: .read),
                .init(tensor: v, access: .read),
                .init(tensor: out, access: .write)
            ]
        }

        public static func route(sq: Int, hd: Int, supportsMPP: Bool) -> Path {
            if sq == 1 { return .decoding }
            if sq >= 32, supportsMPP, hd <= 512 { return .mpp }
            return .plain
        }

        public func encode(to encoder: KernelCommandEncoder) throws {
            switch Self.route(sq: queryLen, hd: headDim, supportsMPP: encoder.device.supportsMPP) {
            case .decoding:
                try encoder.encode(_FlashDecoding(
                    q: q, k: k, v: v, out: out,
                    batch: batch, queryHeads: queryHeads, kvHeads: kvHeads,
                    keyLen: keyLen, headDim: headDim,
                    scale: scale, slidingWindow: slidingWindow
                ))
            case .mpp:
                try encoder.encode(_MPPFlashAttention(
                    q: q, k: k, v: v, out: out,
                    batch: batch, queryHeads: queryHeads, kvHeads: kvHeads,
                    queryLen: queryLen, keyLen: keyLen, headDim: headDim,
                    scale: scale, slidingWindow: slidingWindow
                ))
            case .plain:
                try encoder.dispatch(_FlashAttentionPlain(
                    q: q, k: k, v: v, out: out,
                    batch: batch, queryHeads: queryHeads, kvHeads: kvHeads,
                    queryLen: queryLen, keyLen: keyLen, headDim: headDim,
                    scale: scale, slidingWindow: slidingWindow
                ))
            }
        }

        // MARK: Private

        private let q: Tensor
        private let k: Tensor
        private let v: Tensor
        private let out: Tensor
        private let batch: Int
        private let queryHeads: Int
        private let kvHeads: Int
        private let queryLen: Int
        private let keyLen: Int
        private let headDim: Int
        private let scale: Float
        private let slidingWindow: Int?
    }
}

// MARK: - _FlashAttentionPlain

private struct _FlashAttentionPlain: DispatchKernel {
    // MARK: Lifecycle

    init(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        out: Tensor,
        batch: Int,
        queryHeads: Int,
        kvHeads: Int,
        queryLen: Int,
        keyLen: Int,
        headDim: Int,
        scale: Float,
        slidingWindow: Int? = nil
    ) throws {
        self.q = q
        self.k = k
        self.v = v
        self.out = out
        functionName = switch (q.dataType, k.dataType, out.dataType) {
        case (.f32, .f32, .f32): "flash_attention"
        case (.bf16, .bf16, .bf16): "flash_attention_bf16"
        case (.bf16, .bf16, .f32): "flash_attention_bf16_to_f32"
        case (.f32, .bf16, .bf16): "flash_attention_f32_bf16_to_bf16"
        default:
            throw PopcornError.unsupportedDataTypeCombination(
                "Unsupported FlashAttention data type combination: \(q.dataType), \(k.dataType), \(out.dataType)."
            )
        }
        constants = [FlashAttentionConstants(
            B: UInt32(batch),
            Nq: UInt32(queryHeads),
            Nkv: UInt32(kvHeads),
            Sq: UInt32(queryLen),
            Sk: UInt32(keyLen),
            Hd: UInt32(headDim),
            slidingWindow: Int32(slidingWindow ?? -1),
            scale: scale
        )]
        dispatchGrid = MTLSize(width: batch * Self.tgWidth, height: queryHeads, depth: queryLen)
    }

    // MARK: Internal

    let functionName: String
    let constants: [any BitwiseCopyable]

    var tensors: [Tensor.Binding] {
        [
            .init(tensor: q, access: .read),
            .init(tensor: k, access: .read),
            .init(tensor: v, access: .read),
            .init(tensor: out, access: .write)
        ]
    }

    func dispatchSize(for _: MTLComputePipelineState) -> (grid: MTLSize, threadgroupSize: MTLSize) {
        (dispatchGrid, MTLSize(width: Self.tgWidth, height: 1, depth: 1))
    }

    // MARK: Private

    private static let tgWidth = 128

    private let dispatchGrid: MTLSize
    private let q: Tensor
    private let k: Tensor
    private let v: Tensor
    private let out: Tensor
}

// MARK: - _MPPFlashAttention

private struct _MPPFlashAttention: Kernel {
    // MARK: Lifecycle

    init(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        out: Tensor,
        batch: Int,
        queryHeads: Int,
        kvHeads: Int,
        queryLen: Int,
        keyLen: Int,
        headDim: Int,
        scale: Float,
        slidingWindow: Int? = nil
    ) {
        self.q = q
        self.k = k
        self.v = v
        self.out = out
        self.batch = batch
        self.queryHeads = queryHeads
        self.kvHeads = kvHeads
        self.queryLen = queryLen
        self.keyLen = keyLen
        self.headDim = headDim
        self.scale = scale
        self.slidingWindow = slidingWindow
    }

    // MARK: Internal

    var tensors: [Tensor.Binding] {
        [
            .init(tensor: q, access: .read),
            .init(tensor: k, access: .read),
            .init(tensor: v, access: .read),
            .init(tensor: out, access: .write)
        ]
    }

    func encode(to encoder: KernelCommandEncoder) throws {
        let br = Self.tileBr(for: headDim)
        let maxHd = Self.maxHd(for: headDim)
        guard br > 0, maxHd > 0 else {
            throw PopcornError.tensorShapeMismatch("MPPFlashAttention head dim must be <= 512; got \(headDim).")
        }
        let qTiles = (queryLen + br - 1) / br
        try encoder.withTemporaryTensor(
            .init([batch, queryHeads, qTiles, br, Self.bcTile], .f32),
            .init([batch, queryHeads, qTiles, br, maxHd], .f32)
        ) { sScratch, tScratch in
            try encoder.dispatch(_MPPFAPass(
                q: q, k: k, v: v, out: out,
                sScratch: sScratch, tScratch: tScratch,
                batch: batch, queryHeads: queryHeads, kvHeads: kvHeads,
                queryLen: queryLen, keyLen: keyLen, headDim: headDim,
                scale: scale, slidingWindow: slidingWindow
            ))
        }
    }

    // MARK: Fileprivate

    fileprivate static func tileBr(for headDim: Int) -> Int {
        headDim <= 512 ? 8 : 0
    }

    fileprivate static func maxHd(for headDim: Int) -> Int {
        switch headDim {
        case ...256: 256
        case 257...512: 512
        default: 0
        }
    }

    // MARK: Private

    private static let bcTile = 64

    private let q: Tensor
    private let k: Tensor
    private let v: Tensor
    private let out: Tensor
    private let batch: Int
    private let queryHeads: Int
    private let kvHeads: Int
    private let queryLen: Int
    private let keyLen: Int
    private let headDim: Int
    private let scale: Float
    private let slidingWindow: Int?
}

// MARK: - _MPPFAPass

private struct _MPPFAPass: DispatchKernel {
    // MARK: Lifecycle

    init(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        out: Tensor,
        sScratch: Tensor,
        tScratch: Tensor,
        batch: Int,
        queryHeads: Int,
        kvHeads: Int,
        queryLen: Int,
        keyLen: Int,
        headDim: Int,
        scale: Float,
        slidingWindow: Int? = nil
    ) throws {
        self.q = q
        self.k = k
        self.v = v
        self.out = out
        self.sScratch = sScratch
        self.tScratch = tScratch
        let maxHd = _MPPFlashAttention.maxHd(for: headDim)
        let br = _MPPFlashAttention.tileBr(for: headDim)
        let suffix: String
        switch maxHd {
        case 256: suffix = "hd256"
        case 512: suffix = "hd512"
        default:
            throw PopcornError.tensorShapeMismatch("MPPFlashAttention unsupported head dim bucket \(maxHd).")
        }
        let base: String
        switch (q.dataType, k.dataType, out.dataType) {
        case (.f32, .f32, .f32): base = "mpp_flash_attention_f32"
        case (.bf16, .bf16, .bf16): base = "mpp_flash_attention_bf16"
        case (.bf16, .bf16, .f32): base = "mpp_flash_attention_bf16_to_f32"
        default:
            throw PopcornError.unsupportedDataTypeCombination(
                "Unsupported MPPFlashAttention data type combination: \(q.dataType), \(k.dataType), \(out.dataType)."
            )
        }
        functionName = "\(base)_\(suffix)"
        let qTilesPerHead = (queryLen + br - 1) / br
        constants = [MPPFlashAttentionConstants(
            B: UInt32(batch),
            Nq: UInt32(queryHeads),
            Nkv: UInt32(kvHeads),
            Sq: UInt32(queryLen),
            Sk: UInt32(keyLen),
            Hd: UInt32(headDim),
            qTilesPerHead: UInt32(qTilesPerHead),
            slidingWindow: Int32(slidingWindow ?? -1),
            scale: scale
        )]
        dispatchGrid = MTLSize(width: batch * Self.tgWidth, height: queryHeads, depth: qTilesPerHead)
    }

    // MARK: Internal

    let functionName: String
    let constants: [any BitwiseCopyable]

    var tensors: [Tensor.Binding] {
        [
            .init(tensor: q, access: .read),
            .init(tensor: k, access: .read),
            .init(tensor: v, access: .read),
            .init(tensor: out, access: .write),
            .init(tensor: sScratch, access: .readWrite),
            .init(tensor: tScratch, access: .readWrite)
        ]
    }

    func dispatchSize(for _: MTLComputePipelineState) -> (grid: MTLSize, threadgroupSize: MTLSize) {
        (dispatchGrid, MTLSize(width: Self.tgWidth, height: 1, depth: 1))
    }

    // MARK: Private

    private static let tgWidth = 128

    private let dispatchGrid: MTLSize
    private let q: Tensor
    private let k: Tensor
    private let v: Tensor
    private let out: Tensor
    private let sScratch: Tensor
    private let tScratch: Tensor
}
