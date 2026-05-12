import Metal
import PopcornShaderTypes

// MARK: - _FlashDecoding

struct _FlashDecoding: Kernel {
    // MARK: Lifecycle

    init(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        out: Tensor,
        batch: Int,
        queryHeads: Int,
        kvHeads: Int,
        keyLen: Int,
        headDim: Int,
        scale: Float,
        slidingWindow: Int? = nil
    ) throws {
        guard queryHeads.isMultiple(of: kvHeads) else {
            throw PopcornError.tensorShapeMismatch(
                "FlashDecoding query heads must be divisible by KV heads; got \(queryHeads) and \(kvHeads)."
            )
        }
        guard headDim <= 512 else {
            throw PopcornError.tensorShapeMismatch("FlashDecoding head dim must be <= 512; got \(headDim).")
        }
        self.q = q
        self.k = k
        self.v = v
        self.out = out
        self.batch = batch
        self.queryHeads = queryHeads
        self.kvHeads = kvHeads
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
        try encoder.withTemporaryTensor(
            .init([batch, queryHeads, Self.partitions, headDim], .f32),
            .init([batch, queryHeads, Self.partitions], .f32),
            .init([batch, queryHeads, Self.partitions], .f32)
        ) { partialO, partialM, partialL in
            let partial = try _FDPartial(
                q: q, k: k, v: v,
                partialO: partialO, partialM: partialM, partialL: partialL,
                batch: batch, queryHeads: queryHeads, kvHeads: kvHeads,
                keyLen: keyLen, headDim: headDim, partitions: Self.partitions,
                scale: scale, slidingWindow: slidingWindow
            )
            try encoder.dispatch(partial)
            try encoder.dispatch(_FDReduce(
                partialO: partialO, partialM: partialM, partialL: partialL, out: out,
                batch: batch, queryHeads: queryHeads, headDim: headDim, partitions: Self.partitions
            ))
        }
    }

    // MARK: Private

    private static let partitions = 8

    private let q: Tensor
    private let k: Tensor
    private let v: Tensor
    private let out: Tensor
    private let batch: Int
    private let queryHeads: Int
    private let kvHeads: Int
    private let keyLen: Int
    private let headDim: Int
    private let scale: Float
    private let slidingWindow: Int?
}

// MARK: - _FDPartial

struct _FDPartial: DispatchKernel {
    // MARK: Lifecycle

    init(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        partialO: Tensor,
        partialM: Tensor,
        partialL: Tensor,
        batch: Int,
        queryHeads: Int,
        kvHeads: Int,
        keyLen: Int,
        headDim: Int,
        partitions: Int,
        scale: Float,
        slidingWindow: Int? = nil
    ) throws {
        self.q = q
        self.k = k
        self.v = v
        self.partialO = partialO
        self.partialM = partialM
        self.partialL = partialL
        functionName = switch q.dataType {
        case .f32: "flash_decoding_partial_f32"
        case .bf16: "flash_decoding_partial_bf16"
        default: throw PopcornError.unsupportedDataTypeCombination(
                "Unsupported FlashDecoding Q/K/V data type: \(q.dataType)."
            )
        }
        constants = [FlashDecodingPartialConstants(
            B: UInt32(batch),
            Nq: UInt32(queryHeads),
            Nkv: UInt32(kvHeads),
            Sk: UInt32(keyLen),
            Hd: UInt32(headDim),
            P: UInt32(partitions),
            slidingWindow: Int32(slidingWindow ?? -1),
            scale: scale
        )]
        self.batch = batch
        self.queryHeads = queryHeads
        self.partitions = partitions
    }

    // MARK: Internal

    let functionName: String
    let constants: [any BitwiseCopyable]

    var tensors: [Tensor.Binding] {
        [
            .init(tensor: q, access: .read),
            .init(tensor: k, access: .read),
            .init(tensor: v, access: .read),
            .init(tensor: partialO, access: .write),
            .init(tensor: partialM, access: .write),
            .init(tensor: partialL, access: .write)
        ]
    }

    func dispatchSize(for _: MTLComputePipelineState) -> (grid: MTLSize, threadgroupSize: MTLSize) {
        (
            MTLSize(width: batch * Self.tgWidth, height: queryHeads, depth: partitions),
            MTLSize(width: Self.tgWidth, height: 1, depth: 1)
        )
    }

    // MARK: Private

    private static let tgWidth = 128

    private let q: Tensor
    private let k: Tensor
    private let v: Tensor
    private let partialO: Tensor
    private let partialM: Tensor
    private let partialL: Tensor
    private let batch: Int
    private let queryHeads: Int
    private let partitions: Int
}

// MARK: - _FDReduce

struct _FDReduce: DispatchKernel {
    // MARK: Lifecycle

    init(
        partialO: Tensor,
        partialM: Tensor,
        partialL: Tensor,
        out: Tensor,
        batch: Int,
        queryHeads: Int,
        headDim: Int,
        partitions: Int
    ) throws {
        self.partialO = partialO
        self.partialM = partialM
        self.partialL = partialL
        self.out = out
        functionName = switch out.dataType {
        case .f32: "flash_decoding_reduce_f32"
        case .bf16: "flash_decoding_reduce_bf16"
        default: throw PopcornError.unsupportedDataTypeCombination(
                "Unsupported FlashDecoding output data type: \(out.dataType)."
            )
        }
        constants = [FlashDecodingReduceConstants(
            B: UInt32(batch),
            Nq: UInt32(queryHeads),
            Hd: UInt32(headDim),
            P: UInt32(partitions)
        )]
        self.batch = batch
        self.queryHeads = queryHeads
    }

    // MARK: Internal

    let functionName: String
    let constants: [any BitwiseCopyable]

    var tensors: [Tensor.Binding] {
        [
            .init(tensor: partialO, access: .read),
            .init(tensor: partialM, access: .read),
            .init(tensor: partialL, access: .read),
            .init(tensor: out, access: .write)
        ]
    }

    func dispatchSize(for _: MTLComputePipelineState) -> (grid: MTLSize, threadgroupSize: MTLSize) {
        (
            MTLSize(width: batch * Self.tgWidth, height: queryHeads, depth: 1),
            MTLSize(width: Self.tgWidth, height: 1, depth: 1)
        )
    }

    // MARK: Private

    private static let tgWidth = 128

    private let partialO: Tensor
    private let partialM: Tensor
    private let partialL: Tensor
    private let out: Tensor
    private let batch: Int
    private let queryHeads: Int
}
