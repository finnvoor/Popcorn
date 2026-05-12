import Foundation
import Metal
import MTLSafeTensors
import Popcorn

// MARK: - Gemma4TextInference

final class Gemma4TextInference {
    // MARK: Lifecycle

    init(device: MTLDevice, modelDirectory: URL, maxSeqLen: Int, backend: KernelBackend = .metal4) throws {
        self.maxSeqLen = maxSeqLen
        config = try Gemma4Config.load(from: modelDirectory.appendingPathComponent("config.json"))
        guard maxSeqLen <= config.maxPositionEmbeddings else {
            throw GemmaError.message("maxSeqLen \(maxSeqLen) exceeds model max_position_embeddings \(config.maxPositionEmbeddings).")
        }

        let context = try Self.makeContext(backend: backend, device: device)
        self.context = context
        try Self.preallocateScratch(context, config: config, maxSeqLen: maxSeqLen)

        let archive = try device.makeSafeTensors(from: modelDirectory.appendingPathComponent("model.safetensors"))
        weights = try Gemma4Weights(device: device, archive: archive, layerCount: config.numHiddenLayers)
        workspace = try Workspace(device: device, config: config, maxSeqLen: maxSeqLen)
        rope = try RopeTables(device: device, config: config)

        context.addResidency(weights.allBuffers + workspace.allBuffers + rope.allBuffers)
        context.commitResidency()
    }

    // MARK: Internal

    struct PendingForward {
        // MARK: Internal

        func wait() throws -> Int {
            try feedback.wait()
            let ptr = nextTokenBuffer.contents().bindMemory(to: Int32.self, capacity: 1)
            return Int(ptr[0])
        }

        // MARK: Fileprivate

        fileprivate let feedback: CommitFeedbackBox
        fileprivate let nextTokenBuffer: MTLBuffer
    }

    let config: Gemma4Config

    let context: any KernelContext

    /// Submits the prompt forward pass. Input ids are written to the workspace from the CPU,
    /// and the sampled output token is written into the next ring slot on the GPU.
    func submitPrefill(_ tokens: [Int]) throws -> PendingForward {
        guard !tokens.isEmpty else { throw GemmaError.message("Empty input.") }
        guard tokens.count <= maxSeqLen else {
            throw GemmaError.message("Prompt has \(tokens.count) tokens, exceeding max sequence length \(maxSeqLen).")
        }

        writeTokenIds(tokens)
        writePositions(count: tokens.count, offset: 0)

        let idsTensor = Tensor(buffer: workspace.ids.buffer, shape: [tokens.count], dataType: .i32)
        let positionsTensor = Tensor(buffer: workspace.positions.buffer, shape: [tokens.count], dataType: .i32)
        let outputToken = advanceTokenRing()

        let feedback = try context.submit { encoder in
            try encodeForward(
                tokenCount: tokens.count,
                offset: 0,
                idsTensor: idsTensor,
                positionsTensor: positionsTensor,
                outputToken: outputToken,
                encoder: encoder
            )
        }
        return PendingForward(feedback: feedback, nextTokenBuffer: outputToken.buffer)
    }

    /// Submits a single decode step. The input id is read on the GPU from the previous ring
    /// slot's i32 token buffer, and the position is read from a separate per-step ring slot,
    /// so the call doesn't depend on the previous CB's completion and can be enqueued ahead.
    func submitDecodeStep(at offset: Int) throws -> PendingForward {
        guard offset >= 0, offset < maxSeqLen else {
            throw GemmaError.message("Decode offset \(offset) is out of range [0, \(maxSeqLen)).")
        }

        let inputToken = currentTokenRingSlot()
        let outputToken = advanceTokenRing()
        let positionsTensor = workspace.decodePositionSlots[
            (tokenRingIndex + workspace.decodePositionSlots.count - 1) % workspace.decodePositionSlots.count
        ]
        let ptr = positionsTensor.buffer.contents().bindMemory(to: Int32.self, capacity: 1)
        ptr[0] = Int32(offset)

        let feedback = try context.submit { encoder in
            try encodeForward(
                tokenCount: 1,
                offset: offset,
                idsTensor: inputToken,
                positionsTensor: positionsTensor,
                outputToken: outputToken,
                encoder: encoder
            )
        }
        return PendingForward(feedback: feedback, nextTokenBuffer: outputToken.buffer)
    }

    // MARK: Private

    private let maxSeqLen: Int
    private var tokenRingIndex = 0

    private let weights: Gemma4Weights
    private let workspace: Workspace
    private let rope: RopeTables

    private static func makeContext(backend: KernelBackend, device: MTLDevice) throws -> any KernelContext {
        switch backend {
        case .metal:
            try MetalKernelContext(
                device: device,
                residencyLabel: "Gemma4.residency",
                residencyCapacity: 4096
            )
        case .metal4:
            try Metal4KernelContext(
                device: device,
                residencyLabel: "Gemma4.residency",
                residencyCapacity: 4096,
                constantPageSize: 1 << 20
            )
        }
    }

    private static func preallocateScratch(_ context: any KernelContext, config: Gemma4Config, maxSeqLen: Int) throws {
        let heads = config.numAttentionHeads
        let maxHeadDim = config.globalHeadDim
        let fdPartitions = 8
        let br = 8
        let bc = 64
        let mppMaxHd = maxHeadDim <= 256 ? 256 : 512
        let qTiles = (maxSeqLen + br - 1) / br

        try context.preallocateScratch([
            .init([1, heads, qTiles, br, bc], .f32),
            .init([1, heads, qTiles, br, mppMaxHd], .f32)
        ])
        try context.preallocateScratch([
            .init([1, heads, fdPartitions, maxHeadDim], .f32),
            .init([1, heads, fdPartitions], .f32),
            .init([1, heads, fdPartitions], .f32)
        ])
    }

    private func advanceTokenRing() -> Tensor {
        let slot = workspace.nextTokenSlots[tokenRingIndex % workspace.nextTokenSlots.count]
        tokenRingIndex += 1
        return slot
    }

    private func currentTokenRingSlot() -> Tensor {
        let count = workspace.nextTokenSlots.count
        return workspace.nextTokenSlots[(tokenRingIndex + count - 1) % count]
    }

    private func encodeForward(
        tokenCount t: Int,
        offset: Int,
        idsTensor: Tensor,
        positionsTensor: Tensor,
        outputToken: Tensor,
        encoder: KernelCommandEncoder
    ) throws {
        let h0 = workspace.h0(t)
        let h1 = workspace.h1(t)
        let ids = idsTensor
        let positions = positionsTensor

        try encoder.encode {
            try Kernels.EmbeddingGather(ids: ids, table: weights.embedTokens, into: h0)
            try Kernels.ScalarMul(h0, by: Float(sqrt(Double(config.hiddenSize))), into: h1)
        }
        let inputEmbeds = h1

        try encodePerLayerInputs(ids: ids, inputEmbeds: inputEmbeds, tokenCount: t, encoder: encoder)
        try encodeRopeTables(positions: positions, tokenCount: t, encoder: encoder)

        var currentIsH1 = true
        for layerIndex in 0..<config.numHiddenLayers {
            try encodeLayer(layerIndex, tokenCount: t, offset: offset, currentIsH1: &currentIsH1, encoder: encoder)
        }

        let current = currentIsH1 ? workspace.h1(t) : workspace.h0(t)
        let finalNorm = workspace.normHidden(t)
        // LogitSoftcap (tanh-based) is monotonic, so it doesn't affect argmax. Skipping it
        // saves a 1 MB write/read of cappedLogits per decode step.
        try encoder.encode {
            try Kernels.RMSNorm(current, weight: weights.finalNorm, into: finalNorm, eps: config.rmsNormEps)
            try Kernels.RowSlice2D(finalNorm, into: workspace.lastHidden, rowOffset: t - 1)
            try Kernels.Matmul(workspace.lastHidden, weights.embedTokens, into: workspace.logits, transposeB: true)
            try Kernels.Argmax(workspace.logits, indices: outputToken)
        }
    }

    private func encodePerLayerInputs(ids: Tensor, inputEmbeds: Tensor, tokenCount t: Int, encoder: KernelCommandEncoder) throws {
        let totalColumns = config.numHiddenLayers * config.hiddenSizePerLayerInput
        let pleDim = config.hiddenSizePerLayerInput
        let perLayerRows = t * config.numHiddenLayers

        let tokenRaw = workspace.pleToken.view(shape: [t, totalColumns])
        let tokenScaled = workspace.pleTokenScaled.view(shape: [t, totalColumns])
        let tokenScaledPerLayer = workspace.pleTokenScaled.view(shape: [perLayerRows, pleDim])
        let contextRaw = workspace.pleContext.view(shape: [t, totalColumns])
        let contextScaled = workspace.pleContextScaled.view(shape: [t, totalColumns])
        let contextScaledPerLayer = workspace.pleContextScaled.view(shape: [perLayerRows, pleDim])
        let contextNorm = workspace.pleContextNorm.view(shape: [perLayerRows, pleDim])
        let sumPerLayer = workspace.pleSum.view(shape: [perLayerRows, pleDim])
        let sum = workspace.pleSum.view(shape: [t, totalColumns])
        let full = workspace.pleFull.view(shape: [t, totalColumns])

        try encoder.encode {
            try Kernels.EmbeddingGather(ids: ids, table: weights.embedTokensPerLayer, into: tokenRaw)
            try Kernels.ScalarMul(tokenRaw, by: Float(sqrt(Double(pleDim))), into: tokenScaled)
            try Kernels.Matmul(inputEmbeds, weights.perLayerModelProjection, into: contextRaw, transposeB: true)
            try Kernels.ScalarMul(contextRaw, by: Float(1 / sqrt(Double(config.hiddenSize))), into: contextScaled)
            try Kernels.RMSNorm(contextScaledPerLayer, weight: weights.perLayerProjectionNorm, into: contextNorm, eps: config.rmsNormEps)
            try Kernels.Add(tokenScaledPerLayer, contextNorm, into: sumPerLayer)
            try Kernels.ScalarMul(sum, by: Float(1 / sqrt(2.0)), into: full)
        }
    }

    private func encodeRopeTables(positions: Tensor, tokenCount t: Int, encoder: KernelCommandEncoder) throws {
        let slidingCos = workspace.slidingCos.view(shape: [t, config.headDim / 2])
        let slidingSin = workspace.slidingSin.view(shape: [t, config.headDim / 2])
        let fullCos = workspace.fullCos.view(shape: [t, config.globalHeadDim / 2])
        let fullSin = workspace.fullSin.view(shape: [t, config.globalHeadDim / 2])

        try encoder.encode {
            try Kernels.RopeBuildCosSin(
                positions: positions, invFreq: rope.slidingInvFreq,
                cosOut: slidingCos, sinOut: slidingSin, attentionScaling: 1
            )
            try Kernels.RopeBuildCosSin(
                positions: positions, invFreq: rope.fullInvFreq,
                cosOut: fullCos, sinOut: fullSin, attentionScaling: 1
            )
        }
    }

    private func encodeLayer(
        _ layerIndex: Int,
        tokenCount t: Int,
        offset: Int,
        currentIsH1: inout Bool,
        encoder: KernelCommandEncoder
    ) throws {
        let layer = weights.layers[layerIndex]
        let layerType = config.layerTypes[layerIndex]
        let isSliding = layerType == .sliding
        let headDim = isSliding ? config.headDim : config.globalHeadDim
        let keyLen = offset + t

        let preAttention = currentIsH1 ? workspace.h1(t) : workspace.h0(t)
        let postAttention = currentIsH1 ? workspace.h0(t) : workspace.h1(t)
        try encodeAttention(
            layer: layer,
            layerIndex: layerIndex,
            isSliding: isSliding,
            headDim: headDim,
            tokenCount: t,
            offset: offset,
            keyLen: keyLen,
            input: preAttention,
            output: postAttention,
            encoder: encoder
        )
        currentIsH1.toggle()

        let preMLP = currentIsH1 ? workspace.h1(t) : workspace.h0(t)
        let postMLP = currentIsH1 ? workspace.h0(t) : workspace.h1(t)
        try encodeMLP(
            layer: layer,
            tokenCount: t,
            input: preMLP,
            output: postMLP,
            encoder: encoder
        )
        currentIsH1.toggle()

        let prePLE = currentIsH1 ? workspace.h1(t) : workspace.h0(t)
        let postPLE = currentIsH1 ? workspace.h0(t) : workspace.h1(t)
        try encodePerLayerInputResidual(
            layer: layer,
            layerIndex: layerIndex,
            tokenCount: t,
            input: prePLE,
            output: postPLE,
            encoder: encoder
        )
        currentIsH1.toggle()

        let beforeScalar = currentIsH1 ? workspace.h1(t) : workspace.h0(t)
        let afterScalar = currentIsH1 ? workspace.h0(t) : workspace.h1(t)
        try encoder.encode(Kernels.ScalarMul(beforeScalar, by: layer.layerScalar, into: afterScalar))
        currentIsH1.toggle()
    }

    private func encodeAttention(
        layer: LayerWeights,
        layerIndex: Int,
        isSliding: Bool,
        headDim: Int,
        tokenCount t: Int,
        offset: Int,
        keyLen: Int,
        input: Tensor,
        output: Tensor,
        encoder: KernelCommandEncoder
    ) throws {
        let qWidth = config.numAttentionHeads * headDim
        let kvWidth = config.numKeyValueHeads * headDim

        let normHidden = workspace.normHidden(t)
        let qFlat = workspace.qRaw.view(shape: [t, qWidth])
        let qHeads = workspace.qRaw.view(shape: [1, t, config.numAttentionHeads, headDim])
        let qNormed = workspace.qNorm.view(shape: [1, t, config.numAttentionHeads, headDim])
        let qRoped = workspace.qRope.view(shape: [1, t, config.numAttentionHeads, headDim])
        let qAttn = workspace.qAttn.view(shape: [1, config.numAttentionHeads, t, headDim])

        try encoder.encode {
            try Kernels.RMSNorm(input, weight: layer.inputLayerNorm, into: normHidden, eps: config.rmsNormEps)
            try Kernels.Matmul(normHidden, layer.qProj, into: qFlat, transposeB: true)
        }

        let kvSourceLayer = config.kvSourceLayer(for: layerIndex)
        let ownsKVCache = kvSourceLayer == layerIndex
        if ownsKVCache {
            let kFlat = workspace.kRaw.view(shape: [t, kvWidth])
            let vFlat = workspace.vRaw.view(shape: [t, kvWidth])
            try encoder.encode {
                try Kernels.Matmul(normHidden, layer.kProj!, into: kFlat, transposeB: true)
                try Kernels.Matmul(normHidden, layer.vProj!, into: vFlat, transposeB: true)
            }
        }

        let cos = isSliding
            ? workspace.slidingCos.view(shape: [t, headDim / 2])
            : workspace.fullCos.view(shape: [t, headDim / 2])
        let sin = isSliding
            ? workspace.slidingSin.view(shape: [t, headDim / 2])
            : workspace.fullSin.view(shape: [t, headDim / 2])
        try encoder.encode {
            try Kernels.RMSNorm(qHeads, weight: layer.qNorm, into: qNormed, eps: config.rmsNormEps)
            try Kernels.RopeApply(qNormed, cos: cos, sin: sin, into: qRoped)
            try Kernels.Transpose12(qRoped, into: qAttn)
        }

        if ownsKVCache {
            let kHeads = workspace.kRaw.view(shape: [1, t, config.numKeyValueHeads, headDim])
            let vHeads = workspace.vRaw.view(shape: [1, t, config.numKeyValueHeads, headDim])
            let kNormed = workspace.kNorm.view(shape: [1, t, config.numKeyValueHeads, headDim])
            let vNormed = workspace.vNorm.view(shape: [1, t, config.numKeyValueHeads, headDim])
            let kRoped = workspace.kRope.view(shape: [1, t, config.numKeyValueHeads, headDim])
            let kAttnNew = workspace.kAttnNew.view(shape: [1, config.numKeyValueHeads, t, headDim])
            let vAttnNew = workspace.vAttnNew.view(shape: [1, config.numKeyValueHeads, t, headDim])
            try encoder.encode {
                try Kernels.RMSNorm(kHeads, weight: layer.kNorm!, into: kNormed, eps: config.rmsNormEps)
                try Kernels.RMSNorm(vHeads, weight: nil, into: vNormed, eps: config.rmsNormEps)
                try Kernels.RopeApply(kNormed, cos: cos, sin: sin, into: kRoped)
                try Kernels.Transpose12(kRoped, into: kAttnNew)
                try Kernels.Transpose12(vNormed, into: vAttnNew)
                try Kernels.KVCacheWrite(
                    source: kAttnNew,
                    cache: workspace.kCaches[layerIndex].tensor(headDim: headDim, maxSeqLen: maxSeqLen),
                    offset: offset
                )
                try Kernels.KVCacheWrite(
                    source: vAttnNew,
                    cache: workspace.vCaches[layerIndex].tensor(headDim: headDim, maxSeqLen: maxSeqLen),
                    offset: offset
                )
            }
        }

        let kCache = workspace.kCaches[kvSourceLayer].prefixTensor(headDim: headDim, keyLen: keyLen)
        let vCache = workspace.vCaches[kvSourceLayer].prefixTensor(headDim: headDim, keyLen: keyLen)
        let attnOut = workspace.attnOut.view(shape: [1, config.numAttentionHeads, t, headDim])
        let attnReshaped = workspace.attnOutFlat.view(shape: [1, t, config.numAttentionHeads, headDim])
        let attnFlat = workspace.attnOutFlat.view(shape: [t, qWidth])
        let attnProjected = workspace.attnProjected.view(shape: [t, config.hiddenSize])
        let attnNorm = workspace.attnNorm.view(shape: [t, config.hiddenSize])

        try encoder.encode {
            try Kernels.FlashAttention(
                q: qAttn, k: kCache, v: vCache, into: attnOut,
                scale: 1, slidingWindow: isSliding ? config.slidingWindow : nil
            )
            try Kernels.Transpose12(attnOut, into: attnReshaped)
            try Kernels.Matmul(attnFlat, layer.oProj, into: attnProjected, transposeB: true)
            try Kernels.RMSNorm(attnProjected, weight: layer.postAttentionLayerNorm, into: attnNorm, eps: config.rmsNormEps)
            try Kernels.Add(input, attnNorm, into: output)
        }
    }

    private func encodeMLP(
        layer: LayerWeights,
        tokenCount t: Int,
        input: Tensor,
        output: Tensor,
        encoder: KernelCommandEncoder
    ) throws {
        let normHidden = workspace.normHidden(t)
        let intermediate = layer.intermediateSize
        let gate = workspace.mlpGate.view(shape: [t, intermediate])
        let gateAct = workspace.mlpGateAct.view(shape: [t, intermediate])
        let up = workspace.mlpUp.view(shape: [t, intermediate])
        let gated = workspace.mlpGated.view(shape: [t, intermediate])
        let down = workspace.mlpDown.view(shape: [t, config.hiddenSize])
        let ffnNorm = workspace.ffnNorm.view(shape: [t, config.hiddenSize])

        try encoder.encode {
            try Kernels.RMSNorm(input, weight: layer.preFeedforwardLayerNorm, into: normHidden, eps: config.rmsNormEps)
            try Kernels.Matmul(normHidden, layer.gateProj, into: gate, transposeB: true)
            try Kernels.Matmul(normHidden, layer.upProj, into: up, transposeB: true)
            try Kernels.GeluTanh(gate, into: gateAct)
            try Kernels.Mul(gateAct, up, into: gated)
            try Kernels.Matmul(gated, layer.downProj, into: down, transposeB: true)
            try Kernels.RMSNorm(down, weight: layer.postFeedforwardLayerNorm, into: ffnNorm, eps: config.rmsNormEps)
            try Kernels.Add(input, ffnNorm, into: output)
        }
    }

    private func encodePerLayerInputResidual(
        layer: LayerWeights,
        layerIndex: Int,
        tokenCount t: Int,
        input: Tensor,
        output: Tensor,
        encoder: KernelCommandEncoder
    ) throws {
        let pleDim = config.hiddenSizePerLayerInput
        let pleLayer = workspace.pleLayer.view(shape: [t, pleDim])
        let pleGate = workspace.pleGate.view(shape: [t, pleDim])
        let pleGateAct = workspace.pleGateAct.view(shape: [t, pleDim])
        let pleGated = workspace.pleGated.view(shape: [t, pleDim])
        let pleProjected = workspace.pleProjected.view(shape: [t, config.hiddenSize])
        let pleNorm = workspace.pleNorm.view(shape: [t, config.hiddenSize])
        let pleFull = workspace.pleFull.view(shape: [t, config.numHiddenLayers * pleDim])

        try encoder.encode {
            try Kernels.Slice2D(pleFull, into: pleLayer, columnOffset: layerIndex * pleDim)
            try Kernels.Matmul(input, layer.perLayerInputGate, into: pleGate, transposeB: true)
            try Kernels.GeluTanh(pleGate, into: pleGateAct)
            try Kernels.Mul(pleGateAct, pleLayer, into: pleGated)
            try Kernels.Matmul(pleGated, layer.perLayerProjection, into: pleProjected, transposeB: true)
            try Kernels.RMSNorm(pleProjected, weight: layer.postPerLayerInputNorm, into: pleNorm, eps: config.rmsNormEps)
            try Kernels.Add(input, pleNorm, into: output)
        }
    }

    private func writeTokenIds(_ inputIds: [Int]) {
        let buffer = workspace.ids.buffer
        let ptr = buffer.contents().bindMemory(to: Int32.self, capacity: maxSeqLen)
        for (i, id) in inputIds.enumerated() {
            ptr[i] = Int32(id)
        }
    }

    private func writePositions(count: Int, offset: Int) {
        let buffer = workspace.positions.buffer
        let ptr = buffer.contents().bindMemory(to: Int32.self, capacity: maxSeqLen)
        for i in 0..<count {
            ptr[i] = Int32(offset + i)
        }
    }
}

private extension Tensor {
    func view(shape: [Int]) -> Tensor {
        Tensor(buffer: buffer, shape: shape, dataType: dataType)
    }
}
