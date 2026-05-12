import Foundation
import Metal
import Popcorn

// MARK: - Workspace

final class Workspace {
    // MARK: Lifecycle

    init(device: MTLDevice, config: Gemma4Config, maxSeqLen: Int) throws {
        ids = try device.makeTensor(shape: [maxSeqLen], dataType: .i32, label: "ids")
        positions = try device.makeTensor(shape: [maxSeqLen], dataType: .i32, label: "positions")
        h0 = try device.makeTensor(shape: [maxSeqLen, config.hiddenSize], dataType: .bf16, label: "h0")
        h1 = try device.makeTensor(shape: [maxSeqLen, config.hiddenSize], dataType: .bf16, label: "h1")
        normHidden = try device.makeTensor(shape: [maxSeqLen, config.hiddenSize], dataType: .bf16, label: "normHidden")

        let maxQ = config.numAttentionHeads * config.globalHeadDim
        let maxKV = config.numKeyValueHeads * config.globalHeadDim
        qRaw = try device.makeTensor(shape: [maxSeqLen, maxQ], dataType: .bf16, label: "qRaw")
        kRaw = try device.makeTensor(shape: [maxSeqLen, maxKV], dataType: .bf16, label: "kRaw")
        vRaw = try device.makeTensor(shape: [maxSeqLen, maxKV], dataType: .bf16, label: "vRaw")
        qNorm = try device.makeTensor(shape: [maxSeqLen, maxQ], dataType: .bf16, label: "qNorm")
        kNorm = try device.makeTensor(shape: [maxSeqLen, maxKV], dataType: .bf16, label: "kNorm")
        vNorm = try device.makeTensor(shape: [maxSeqLen, maxKV], dataType: .bf16, label: "vNorm")
        qRope = try device.makeTensor(shape: [maxSeqLen, maxQ], dataType: .bf16, label: "qRope")
        kRope = try device.makeTensor(shape: [maxSeqLen, maxKV], dataType: .bf16, label: "kRope")
        qAttn = try device.makeTensor(shape: [1, config.numAttentionHeads, maxSeqLen, config.globalHeadDim], dataType: .bf16, label: "qAttn")
        kAttnNew = try device.makeTensor(shape: [1, config.numKeyValueHeads, maxSeqLen, config.globalHeadDim], dataType: .bf16, label: "kAttnNew")
        vAttnNew = try device.makeTensor(shape: [1, config.numKeyValueHeads, maxSeqLen, config.globalHeadDim], dataType: .bf16, label: "vAttnNew")

        attnOut = try device.makeTensor(shape: [1, config.numAttentionHeads, maxSeqLen, config.globalHeadDim], dataType: .bf16, label: "attnOut")
        attnOutFlat = try device.makeTensor(shape: [maxSeqLen, maxQ], dataType: .bf16, label: "attnOutFlat")
        attnProjected = try device.makeTensor(shape: [maxSeqLen, config.hiddenSize], dataType: .bf16, label: "attnProjected")
        attnNorm = try device.makeTensor(shape: [maxSeqLen, config.hiddenSize], dataType: .bf16, label: "attnNorm")

        let maxIntermediate = config.intermediateSize * 2
        mlpGate = try device.makeTensor(shape: [maxSeqLen, maxIntermediate], dataType: .bf16, label: "mlpGate")
        mlpGateAct = try device.makeTensor(shape: [maxSeqLen, maxIntermediate], dataType: .bf16, label: "mlpGateAct")
        mlpUp = try device.makeTensor(shape: [maxSeqLen, maxIntermediate], dataType: .bf16, label: "mlpUp")
        mlpGated = try device.makeTensor(shape: [maxSeqLen, maxIntermediate], dataType: .bf16, label: "mlpGated")
        mlpDown = try device.makeTensor(shape: [maxSeqLen, config.hiddenSize], dataType: .bf16, label: "mlpDown")
        ffnNorm = try device.makeTensor(shape: [maxSeqLen, config.hiddenSize], dataType: .bf16, label: "ffnNorm")

        let pleColumns = config.numHiddenLayers * config.hiddenSizePerLayerInput
        pleToken = try device.makeTensor(shape: [maxSeqLen, pleColumns], dataType: .bf16, label: "pleToken")
        pleTokenScaled = try device.makeTensor(shape: [maxSeqLen, pleColumns], dataType: .bf16, label: "pleTokenScaled")
        pleContext = try device.makeTensor(shape: [maxSeqLen, pleColumns], dataType: .bf16, label: "pleContext")
        pleContextScaled = try device.makeTensor(shape: [maxSeqLen, pleColumns], dataType: .bf16, label: "pleContextScaled")
        pleContextNorm = try device.makeTensor(shape: [maxSeqLen, pleColumns], dataType: .bf16, label: "pleContextNorm")
        pleSum = try device.makeTensor(shape: [maxSeqLen, pleColumns], dataType: .bf16, label: "pleSum")
        pleFull = try device.makeTensor(shape: [maxSeqLen, pleColumns], dataType: .bf16, label: "pleFull")
        pleLayer = try device.makeTensor(shape: [maxSeqLen, config.hiddenSizePerLayerInput], dataType: .bf16, label: "pleLayer")
        pleGate = try device.makeTensor(shape: [maxSeqLen, config.hiddenSizePerLayerInput], dataType: .bf16, label: "pleGate")
        pleGateAct = try device.makeTensor(shape: [maxSeqLen, config.hiddenSizePerLayerInput], dataType: .bf16, label: "pleGateAct")
        pleGated = try device.makeTensor(shape: [maxSeqLen, config.hiddenSizePerLayerInput], dataType: .bf16, label: "pleGated")
        pleProjected = try device.makeTensor(shape: [maxSeqLen, config.hiddenSize], dataType: .bf16, label: "pleProjected")
        pleNorm = try device.makeTensor(shape: [maxSeqLen, config.hiddenSize], dataType: .bf16, label: "pleNorm")

        slidingCos = try device.makeTensor(shape: [maxSeqLen, config.headDim / 2], dataType: .f32, label: "slidingCos")
        slidingSin = try device.makeTensor(shape: [maxSeqLen, config.headDim / 2], dataType: .f32, label: "slidingSin")
        fullCos = try device.makeTensor(shape: [maxSeqLen, config.globalHeadDim / 2], dataType: .f32, label: "fullCos")
        fullSin = try device.makeTensor(shape: [maxSeqLen, config.globalHeadDim / 2], dataType: .f32, label: "fullSin")

        lastHidden = try device.makeTensor(shape: [1, config.hiddenSize], dataType: .bf16, label: "lastHidden")
        logits = try device.makeTensor(shape: [1, config.vocabSize], dataType: .f32, label: "logits")
        cappedLogits = try device.makeTensor(shape: [1, config.vocabSize], dataType: .f32, label: "cappedLogits")
        nextTokenSlots = try (0..<Workspace.tokenRingCapacity).map { i in
            try device.makeTensor(shape: [1], dataType: .i32, label: "nextTokenSlot.\(i)")
        }
        decodePositionSlots = try (0..<Workspace.tokenRingCapacity).map { i in
            try device.makeTensor(shape: [1], dataType: .i32, label: "decodePositionSlot.\(i)")
        }

        kCaches = try (0..<config.numHiddenLayers).map { i in
            let headDim = config.layerTypes[i] == .sliding ? config.headDim : config.globalHeadDim
            return try CacheTensor(device: device, maxSeqLen: maxSeqLen, headDim: headDim, label: "kCache.\(i)")
        }
        vCaches = try (0..<config.numHiddenLayers).map { i in
            let headDim = config.layerTypes[i] == .sliding ? config.headDim : config.globalHeadDim
            return try CacheTensor(device: device, maxSeqLen: maxSeqLen, headDim: headDim, label: "vCache.\(i)")
        }
    }

    // MARK: Internal

    static let tokenRingCapacity = 4

    let ids: Tensor
    let positions: Tensor

    let h0: Tensor
    let h1: Tensor

    let normHidden: Tensor

    let qRaw: Tensor; let kRaw: Tensor; let vRaw: Tensor
    let qNorm: Tensor; let kNorm: Tensor; let vNorm: Tensor
    let qRope: Tensor; let kRope: Tensor
    let qAttn: Tensor; let kAttnNew: Tensor; let vAttnNew: Tensor
    let attnOut: Tensor
    let attnOutFlat: Tensor
    let attnProjected: Tensor
    let attnNorm: Tensor

    let mlpGate: Tensor; let mlpGateAct: Tensor; let mlpUp: Tensor; let mlpGated: Tensor
    let mlpDown: Tensor; let ffnNorm: Tensor

    let pleToken: Tensor; let pleTokenScaled: Tensor
    let pleContext: Tensor; let pleContextScaled: Tensor; let pleContextNorm: Tensor
    let pleSum: Tensor; let pleFull: Tensor
    let pleLayer: Tensor
    let pleGate: Tensor; let pleGateAct: Tensor; let pleGated: Tensor
    let pleProjected: Tensor; let pleNorm: Tensor

    let slidingCos: Tensor; let slidingSin: Tensor
    let fullCos: Tensor; let fullSin: Tensor

    let lastHidden: Tensor
    let logits: Tensor

    let cappedLogits: Tensor
    let nextTokenSlots: [Tensor]
    let decodePositionSlots: [Tensor]

    let kCaches: [CacheTensor]
    let vCaches: [CacheTensor]

    var allBuffers: [MTLBuffer] {
        var buffers: [MTLBuffer] = [
            ids.buffer, positions.buffer, h0.buffer, h1.buffer, normHidden.buffer,
            qRaw.buffer, kRaw.buffer, vRaw.buffer,
            qNorm.buffer, kNorm.buffer, vNorm.buffer,
            qRope.buffer, kRope.buffer, qAttn.buffer, kAttnNew.buffer, vAttnNew.buffer,
            attnOut.buffer, attnOutFlat.buffer, attnProjected.buffer, attnNorm.buffer,
            mlpGate.buffer, mlpGateAct.buffer, mlpUp.buffer, mlpGated.buffer, mlpDown.buffer, ffnNorm.buffer,
            pleToken.buffer, pleTokenScaled.buffer, pleContext.buffer, pleContextScaled.buffer,
            pleContextNorm.buffer, pleSum.buffer, pleFull.buffer,
            pleLayer.buffer, pleGate.buffer, pleGateAct.buffer, pleGated.buffer,
            pleProjected.buffer, pleNorm.buffer,
            slidingCos.buffer, slidingSin.buffer, fullCos.buffer, fullSin.buffer,
            lastHidden.buffer, logits.buffer, cappedLogits.buffer
        ]
        buffers.append(contentsOf: nextTokenSlots.map(\.buffer))
        buffers.append(contentsOf: decodePositionSlots.map(\.buffer))
        for cache in kCaches {
            buffers.append(cache.storage.buffer)
        }
        for cache in vCaches {
            buffers.append(cache.storage.buffer)
        }
        return buffers
    }

    func h0(_ t: Int) -> Tensor {
        Tensor(buffer: h0.buffer, shape: [t, h0.shape[1]], dataType: .bf16)
    }

    func h1(_ t: Int) -> Tensor {
        Tensor(buffer: h1.buffer, shape: [t, h1.shape[1]], dataType: .bf16)
    }

    func normHidden(_ t: Int) -> Tensor {
        Tensor(buffer: normHidden.buffer, shape: [t, h0.shape[1]], dataType: .bf16)
    }
}

// MARK: - CacheTensor

struct CacheTensor {
    // MARK: Lifecycle

    init(device: MTLDevice, maxSeqLen: Int, headDim: Int, label: String) throws {
        storage = try device.makeTensor(shape: [1, 1, maxSeqLen, headDim], dataType: .bf16, label: label)
    }

    // MARK: Internal

    let storage: Tensor

    func tensor(headDim: Int, maxSeqLen: Int) -> Tensor {
        Tensor(buffer: storage.buffer, shape: [1, 1, maxSeqLen, headDim], dataType: .bf16)
    }

    func prefixTensor(headDim: Int, keyLen: Int) -> Tensor {
        Tensor(buffer: storage.buffer, shape: [1, 1, keyLen, headDim], dataType: .bf16)
    }
}

// MARK: - RopeTables

struct RopeTables {
    // MARK: Lifecycle

    init(device: MTLDevice, config: Gemma4Config) throws {
        let sliding = (0..<(config.headDim / 2)).map { i in
            Float(1 / pow(10_000, Double(2 * i) / Double(config.headDim)))
        }
        let rotated = config.globalHeadDim / 8
        let full = (0..<(config.globalHeadDim / 2)).map { i -> Float in
            i < rotated ? Float(1 / pow(1_000_000, Double(2 * i) / Double(config.globalHeadDim))) : 0
        }
        slidingInvFreq = try device.makeTensor(values: sliding, shape: [config.headDim / 2], label: "rope.sliding.inv_freq")
        fullInvFreq = try device.makeTensor(values: full, shape: [config.globalHeadDim / 2], label: "rope.full.inv_freq")
    }

    // MARK: Internal

    let slidingInvFreq: Tensor
    let fullInvFreq: Tensor

    var allBuffers: [MTLBuffer] {
        [slidingInvFreq.buffer, fullInvFreq.buffer]
    }
}
