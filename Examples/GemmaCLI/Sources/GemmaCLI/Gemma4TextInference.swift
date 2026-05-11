import Foundation
import Metal
import MTLSafeTensors
import Popcorn

private func secs(_ a: ContinuousClock.Instant, _ b: ContinuousClock.Instant) -> Double {
    let d = b - a
    return Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
}

// MARK: - Gemma4TextInference

final class Gemma4TextInference {
    // MARK: Lifecycle

    init(device: MTLDevice, modelDirectory: URL, maxSeqLen: Int) throws {
        self.device = device
        self.maxSeqLen = maxSeqLen
        config = try Gemma4Config.load(from: modelDirectory.appendingPathComponent("config.json"))
        guard maxSeqLen <= config.maxPositionEmbeddings else {
            throw GemmaError.message("maxSeqLen \(maxSeqLen) exceeds model max_position_embeddings \(config.maxPositionEmbeddings).")
        }

        kernelLibrary = try KernelLibrary(device: device)
        guard let queue = device.makeMTL4CommandQueue() else {
            throw GemmaError.message("Could not create MTL4CommandQueue.")
        }
        commandQueue = queue
        guard let allocator = device.makeCommandAllocator() else {
            throw GemmaError.message("Could not create MTL4CommandAllocator.")
        }
        commandAllocator = allocator

        let residencyDescriptor = MTLResidencySetDescriptor()
        residencyDescriptor.label = "Gemma4.residency"
        residencyDescriptor.initialCapacity = 4096
        residencySet = try device.makeResidencySet(descriptor: residencyDescriptor)

        argumentTables = ArgumentTablePool(device: device)

        constants = Metal4ConstantArena(device: device, residencySet: residencySet, pageSize: 1 << 20)
        try constants.preallocate(pageCount: 1)

        archive = try device.makeSafeTensors(from: modelDirectory.appendingPathComponent("model.safetensors"))
        weights = try Gemma4Weights(device: device, archive: archive, layerCount: config.numHiddenLayers)
        workspace = try Workspace(device: device, config: config, maxSeqLen: maxSeqLen)
        rope = try RopeTables(device: device, config: config)

        for buffer in weights.allBuffers + workspace.allBuffers + rope.allBuffers {
            residencySet.addAllocation(buffer)
        }
        residencySet.commit()
        residencySet.requestResidency()
        commandQueue.addResidencySet(residencySet)
    }

    // MARK: Internal

    struct PendingForward {
        // MARK: Internal

        func wait() throws -> Int {
            try feedback.wait()
            let t = ContinuousClock.now
            Gemma4TextInference.debugCommitWaitSeconds += secs(encodeEnd, t)
            let ptr = nextTokenBuffer.contents().bindMemory(to: Int32.self, capacity: 1)
            return Int(ptr[0])
        }

        // MARK: Fileprivate

        fileprivate let feedback: CommitFeedbackBox
        fileprivate let nextTokenBuffer: MTLBuffer
        fileprivate let encodeEnd: ContinuousClock.Instant
    }

    let config: Gemma4Config

    func nextToken(inputIds: [Int], offset: Int) throws -> Int {
        try submit(inputIds: inputIds, offset: offset).wait()
    }

    nonisolated(unsafe) static var debugEncodeSeconds: Double = 0
    nonisolated(unsafe) static var debugCommitWaitSeconds: Double = 0
    nonisolated(unsafe) static var debugGPUSeconds: Double = 0
    nonisolated(unsafe) static var debugCallCount: Int = 0

    func submit(inputIds: [Int], offset: Int) throws -> PendingForward {
        guard !inputIds.isEmpty else { throw GemmaError.message("Empty input.") }
        guard offset >= 0, offset + inputIds.count <= maxSeqLen else {
            throw GemmaError.message(
                "Token range \(offset)..<\(offset + inputIds.count) exceeds max sequence length \(maxSeqLen)."
            )
        }

        let t0 = ContinuousClock.now
        writeTokenIds(inputIds)
        writePositions(count: inputIds.count, offset: offset)

        guard let commandBuffer = device.makeCommandBuffer() else {
            throw GemmaError.message("Could not create MTL4CommandBuffer.")
        }
        commandBuffer.beginCommandBuffer(allocator: commandAllocator)
        guard let compute = commandBuffer.makeComputeCommandEncoder() else {
            throw GemmaError.message("Could not create MTL4ComputeCommandEncoder.")
        }
        argumentTables.reset()

        let encoder = HazardTrackingEncoder(
            encoder: compute,
            kernelLibrary: kernelLibrary,
            argumentTables: argumentTables,
            constants: constants
        )
        try encodeForward(tokenCount: inputIds.count, offset: offset, encoder: encoder)
        compute.endEncoding()
        commandBuffer.endCommandBuffer()

        let feedback = CommitFeedbackBox()
        let options = MTL4CommitOptions()
        options.addFeedbackHandler { [constants] commitFeedback in
            constants.reset()
            let gpu = commitFeedback.gpuEndTime - commitFeedback.gpuStartTime
            if gpu > 0 { Gemma4TextInference.debugGPUSeconds += gpu }
            feedback.finish(error: commitFeedback.error)
        }
        commandQueue.commit([commandBuffer], options: options)
        let t1 = ContinuousClock.now
        Self.debugEncodeSeconds += secs(t0, t1)
        Self.debugCallCount += 1

        return PendingForward(feedback: feedback, nextTokenBuffer: workspace.nextToken.buffer, encodeEnd: t1)
    }

    // MARK: Private

    private let device: MTLDevice
    private let maxSeqLen: Int
    private let kernelLibrary: KernelLibrary
    private let commandQueue: any MTL4CommandQueue
    private let commandAllocator: any MTL4CommandAllocator
    private let residencySet: any MTLResidencySet
    private let argumentTables: ArgumentTablePool
    private let constants: Metal4ConstantArena

    private let archive: SafeTensors
    private let weights: Gemma4Weights
    private let workspace: Workspace
    private let rope: RopeTables

    private func encodeForward(tokenCount t: Int, offset: Int, encoder: HazardTrackingEncoder) throws {
        let h0 = workspace.h0(t)
        let h1 = workspace.h1(t)
        let ids = Tensor(buffer: workspace.ids.buffer, shape: [t], dataType: .i32)
        let positions = Tensor(buffer: workspace.positions.buffer, shape: [t], dataType: .i32)

        try encoder.encode(Kernels.EmbeddingGather(ids: ids, table: weights.embedTokens, into: h0))
        try encoder.encode(Kernels.ScalarMul(h0, by: Float(sqrt(Double(config.hiddenSize))), into: h1))
        let inputEmbeds = h1

        try encodePerLayerInputs(ids: ids, inputEmbeds: inputEmbeds, tokenCount: t, encoder: encoder)
        try encodeRopeTables(positions: positions, tokenCount: t, encoder: encoder)

        var currentIsH1 = true
        for layerIndex in 0..<config.numHiddenLayers {
            try encodeLayer(layerIndex, tokenCount: t, offset: offset, currentIsH1: &currentIsH1, encoder: encoder)
        }

        let current = currentIsH1 ? workspace.h1(t) : workspace.h0(t)
        let finalNorm = workspace.normHidden(t)
        try encoder.encode(Kernels.RMSNorm(current, weight: weights.finalNorm, into: finalNorm, eps: config.rmsNormEps))
        try encoder.encode(Kernels.RowSlice2D(finalNorm, into: workspace.lastHidden, rowOffset: t - 1))
        try encoder.encode(Kernels.Matmul(workspace.lastHidden, weights.embedTokens, into: workspace.logits, transposeB: true))
        try encoder.encode(Kernels.LogitSoftcap(workspace.logits, cap: config.finalLogitSoftcap, into: workspace.cappedLogits))
        try encoder.encode(Kernels.Argmax(workspace.cappedLogits, indices: workspace.nextToken))
    }

    private func encodePerLayerInputs(ids: Tensor, inputEmbeds: Tensor, tokenCount t: Int, encoder: HazardTrackingEncoder) throws {
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

        try encoder.encode(Kernels.EmbeddingGather(ids: ids, table: weights.embedTokensPerLayer, into: tokenRaw))
        try encoder.encode(Kernels.ScalarMul(tokenRaw, by: Float(sqrt(Double(pleDim))), into: tokenScaled))
        try encoder.encode(Kernels.Matmul(inputEmbeds, weights.perLayerModelProjection, into: contextRaw, transposeB: true))
        try encoder.encode(Kernels.ScalarMul(contextRaw, by: Float(1 / sqrt(Double(config.hiddenSize))), into: contextScaled))
        try encoder.encode(Kernels.RMSNorm(contextScaledPerLayer, weight: weights.perLayerProjectionNorm, into: contextNorm, eps: config.rmsNormEps))
        try encoder.encode(Kernels.Add(tokenScaledPerLayer, contextNorm, into: sumPerLayer))
        try encoder.encode(Kernels.ScalarMul(sum, by: Float(1 / sqrt(2.0)), into: full))
    }

    private func encodeRopeTables(positions: Tensor, tokenCount t: Int, encoder: HazardTrackingEncoder) throws {
        let slidingCos = workspace.slidingCos.view(shape: [t, config.headDim / 2])
        let slidingSin = workspace.slidingSin.view(shape: [t, config.headDim / 2])
        let fullCos = workspace.fullCos.view(shape: [t, config.globalHeadDim / 2])
        let fullSin = workspace.fullSin.view(shape: [t, config.globalHeadDim / 2])

        try encoder.encode(Kernels.RopeBuildCosSin(
            positions: positions, invFreq: rope.slidingInvFreq,
            cosOut: slidingCos, sinOut: slidingSin, attentionScaling: 1
        ))
        try encoder.encode(Kernels.RopeBuildCosSin(
            positions: positions, invFreq: rope.fullInvFreq,
            cosOut: fullCos, sinOut: fullSin, attentionScaling: 1
        ))
    }

    private func encodeLayer(
        _ layerIndex: Int,
        tokenCount t: Int,
        offset: Int,
        currentIsH1: inout Bool,
        encoder: HazardTrackingEncoder
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
        encoder: HazardTrackingEncoder
    ) throws {
        let qWidth = config.numAttentionHeads * headDim
        let kvWidth = config.numKeyValueHeads * headDim

        let normHidden = workspace.normHidden(t)
        try encoder.encode(Kernels.RMSNorm(input, weight: layer.inputLayerNorm, into: normHidden, eps: config.rmsNormEps))

        let qFlat = workspace.qRaw.view(shape: [t, qWidth])
        let qHeads = workspace.qRaw.view(shape: [1, t, config.numAttentionHeads, headDim])
        let qNormed = workspace.qNorm.view(shape: [1, t, config.numAttentionHeads, headDim])
        let qRoped = workspace.qRope.view(shape: [1, t, config.numAttentionHeads, headDim])
        let qAttn = workspace.qAttn.view(shape: [1, config.numAttentionHeads, t, headDim])
        try encoder.encode(Kernels.Matmul(normHidden, layer.qProj, into: qFlat, transposeB: true))

        let kvSourceLayer = config.kvSourceLayer(for: layerIndex)
        let ownsKVCache = kvSourceLayer == layerIndex
        if ownsKVCache {
            let kFlat = workspace.kRaw.view(shape: [t, kvWidth])
            let vFlat = workspace.vRaw.view(shape: [t, kvWidth])
            try encoder.encode(Kernels.Matmul(normHidden, layer.kProj!, into: kFlat, transposeB: true))
            try encoder.encode(Kernels.Matmul(normHidden, layer.vProj!, into: vFlat, transposeB: true))
        }

        try encoder.encode(Kernels.RMSNorm(qHeads, weight: layer.qNorm, into: qNormed, eps: config.rmsNormEps))
        let cos = isSliding
            ? workspace.slidingCos.view(shape: [t, headDim / 2])
            : workspace.fullCos.view(shape: [t, headDim / 2])
        let sin = isSliding
            ? workspace.slidingSin.view(shape: [t, headDim / 2])
            : workspace.fullSin.view(shape: [t, headDim / 2])
        try encoder.encode(Kernels.RopeApply(qNormed, cos: cos, sin: sin, into: qRoped))
        let qAttnInput: Tensor
        if t == 1 {
            // Transpose12 of [B, T=1, Nh, Hd] -> [B, Nh, T=1, Hd] is a no-op:
            // both layouts have the exact same memory order when T=1, so we
            // can just re-view qRoped under the destination shape and skip the
            // dispatch entirely.
            qAttnInput = workspace.qRope.view(shape: [1, config.numAttentionHeads, 1, headDim])
        } else {
            try encoder.encode(Kernels.Transpose12(qRoped, into: qAttn))
            qAttnInput = qAttn
        }

        if ownsKVCache {
            let kHeads = workspace.kRaw.view(shape: [1, t, config.numKeyValueHeads, headDim])
            let vHeads = workspace.vRaw.view(shape: [1, t, config.numKeyValueHeads, headDim])
            let kNormed = workspace.kNorm.view(shape: [1, t, config.numKeyValueHeads, headDim])
            let vNormed = workspace.vNorm.view(shape: [1, t, config.numKeyValueHeads, headDim])
            let kRoped = workspace.kRope.view(shape: [1, t, config.numKeyValueHeads, headDim])
            let kAttnNew = workspace.kAttnNew.view(shape: [1, config.numKeyValueHeads, t, headDim])
            let vAttnNew = workspace.vAttnNew.view(shape: [1, config.numKeyValueHeads, t, headDim])
            try encoder.encode(Kernels.RMSNorm(kHeads, weight: layer.kNorm!, into: kNormed, eps: config.rmsNormEps))

            try encoder.encode(Kernels.RMSNorm(vHeads, weight: nil, into: vNormed, eps: config.rmsNormEps))
            try encoder.encode(Kernels.RopeApply(kNormed, cos: cos, sin: sin, into: kRoped))
            let kForCache: Tensor
            let vForCache: Tensor
            if t == 1 {
                // Same no-op transpose: re-view directly under [B, Nkv, T, Hd].
                kForCache = workspace.kRope.view(shape: [1, config.numKeyValueHeads, 1, headDim])
                vForCache = workspace.vNorm.view(shape: [1, config.numKeyValueHeads, 1, headDim])
            } else {
                try encoder.encode(Kernels.Transpose12(kRoped, into: kAttnNew))
                try encoder.encode(Kernels.Transpose12(vNormed, into: vAttnNew))
                kForCache = kAttnNew
                vForCache = vAttnNew
            }
            try encoder.encode(Kernels.KVCacheWrite(
                source: kForCache,
                cache: workspace.kCaches[layerIndex].tensor(headDim: headDim, maxSeqLen: maxSeqLen),
                offset: offset
            ))
            try encoder.encode(Kernels.KVCacheWrite(
                source: vForCache,
                cache: workspace.vCaches[layerIndex].tensor(headDim: headDim, maxSeqLen: maxSeqLen),
                offset: offset
            ))
        }

        let kCache = workspace.kCaches[kvSourceLayer].prefixTensor(headDim: headDim, keyLen: keyLen)
        let vCache = workspace.vCaches[kvSourceLayer].prefixTensor(headDim: headDim, keyLen: keyLen)
        let probs = workspace.attnProbs.view(shape: [1, config.numAttentionHeads, t, keyLen])
        let attnOut = workspace.attnOut.view(shape: [1, config.numAttentionHeads, t, headDim])
        let attnReshaped = workspace.attnOutFlat.view(shape: [1, t, config.numAttentionHeads, headDim])
        let attnFlat = workspace.attnOutFlat.view(shape: [t, qWidth])
        let attnProjected = workspace.attnProjected.view(shape: [t, config.hiddenSize])
        let attnNorm = workspace.attnNorm.view(shape: [t, config.hiddenSize])

        try encoder.encode(Kernels.AttentionScoresSoftmax(
            q: qAttnInput, k: kCache, into: probs,
            scale: 1, slidingWindow: isSliding ? config.slidingWindow : nil
        ))
        // AttentionOutput writes [B, Nq, t, Hd]. The subsequent Transpose12 to
        // [B, t, Nq, Hd] is a no-op when t=1, so we just point the matmul at
        // the attn output buffer directly.
        let attnFlatInput: Tensor
        if t == 1 {
            try encoder.encode(Kernels.AttentionOutput(scores: probs, v: vCache, into: attnOut))
            attnFlatInput = workspace.attnOut.view(shape: [t, qWidth])
        } else {
            try encoder.encode(Kernels.AttentionOutput(scores: probs, v: vCache, into: attnOut))
            try encoder.encode(Kernels.Transpose12(attnOut, into: attnReshaped))
            attnFlatInput = attnFlat
        }
        try encoder.encode(Kernels.Matmul(attnFlatInput, layer.oProj, into: attnProjected, transposeB: true))
        try encoder.encode(Kernels.RMSNorm(attnProjected, weight: layer.postAttentionLayerNorm, into: attnNorm, eps: config.rmsNormEps))
        try encoder.encode(Kernels.Add(input, attnNorm, into: output))
    }

    private func encodeMLP(
        layer: LayerWeights,
        tokenCount t: Int,
        input: Tensor,
        output: Tensor,
        encoder: HazardTrackingEncoder
    ) throws {
        let normHidden = workspace.normHidden(t)
        try encoder.encode(Kernels.RMSNorm(input, weight: layer.preFeedforwardLayerNorm, into: normHidden, eps: config.rmsNormEps))

        let intermediate = layer.intermediateSize
        let gate = workspace.mlpGate.view(shape: [t, intermediate])
        let gateAct = workspace.mlpGateAct.view(shape: [t, intermediate])
        let up = workspace.mlpUp.view(shape: [t, intermediate])
        let gated = workspace.mlpGated.view(shape: [t, intermediate])
        let down = workspace.mlpDown.view(shape: [t, config.hiddenSize])
        let ffnNorm = workspace.ffnNorm.view(shape: [t, config.hiddenSize])

        if t == 1, Kernels.SwigluMatvec.supports(x: normHidden.dataType, w: layer.gateProj.dataType, o: gated.dataType) {
            try encoder.encode(Kernels.SwigluMatvec(x: normHidden, gate: layer.gateProj, up: layer.upProj, out: gated))
        } else {
            try encoder.encode(Kernels.Matmul(normHidden, layer.gateProj, into: gate, transposeB: true))
            try encoder.encode(Kernels.Matmul(normHidden, layer.upProj, into: up, transposeB: true))
            try encoder.encode(Kernels.GeluTanh(gate, into: gateAct))
            try encoder.encode(Kernels.Mul(gateAct, up, into: gated))
        }
        try encoder.encode(Kernels.Matmul(gated, layer.downProj, into: down, transposeB: true))
        try encoder.encode(Kernels.RMSNorm(down, weight: layer.postFeedforwardLayerNorm, into: ffnNorm, eps: config.rmsNormEps))
        try encoder.encode(Kernels.Add(input, ffnNorm, into: output))
    }

    private func encodePerLayerInputResidual(
        layer: LayerWeights,
        layerIndex: Int,
        tokenCount t: Int,
        input: Tensor,
        output: Tensor,
        encoder: HazardTrackingEncoder
    ) throws {
        let pleDim = config.hiddenSizePerLayerInput
        let pleLayer = workspace.pleLayer.view(shape: [t, pleDim])
        let pleGate = workspace.pleGate.view(shape: [t, pleDim])
        let pleGateAct = workspace.pleGateAct.view(shape: [t, pleDim])
        let pleGated = workspace.pleGated.view(shape: [t, pleDim])
        let pleProjected = workspace.pleProjected.view(shape: [t, config.hiddenSize])
        let pleNorm = workspace.pleNorm.view(shape: [t, config.hiddenSize])
        let pleFull = workspace.pleFull.view(shape: [t, config.numHiddenLayers * pleDim])

        try encoder.encode(Kernels.Slice2D(pleFull, into: pleLayer, columnOffset: layerIndex * pleDim))
        try encoder.encode(Kernels.Matmul(input, layer.perLayerInputGate, into: pleGate, transposeB: true))
        try encoder.encode(Kernels.GeluTanh(pleGate, into: pleGateAct))
        try encoder.encode(Kernels.Mul(pleGateAct, pleLayer, into: pleGated))
        try encoder.encode(Kernels.Matmul(pleGated, layer.perLayerProjection, into: pleProjected, transposeB: true))
        try encoder.encode(Kernels.RMSNorm(pleProjected, weight: layer.postPerLayerInputNorm, into: pleNorm, eps: config.rmsNormEps))
        try encoder.encode(Kernels.Add(input, pleNorm, into: output))
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
