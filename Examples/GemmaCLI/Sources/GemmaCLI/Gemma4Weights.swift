import Metal
import MTLSafeTensors
import Popcorn

// MARK: - Gemma4Weights

struct Gemma4Weights {
    // MARK: Lifecycle

    init(device: MTLDevice, archive: SafeTensors, layerCount: Int) throws {
        embedTokens = try archive.popcornTensor("model.language_model.embed_tokens.weight")
        embedTokensPerLayer = try archive.popcornTensor("model.language_model.embed_tokens_per_layer.weight")
        finalNorm = try device.makeNormWeightTensor(archive: archive, name: "model.language_model.norm.weight")
        perLayerModelProjection = try archive.popcornTensor("model.language_model.per_layer_model_projection.weight")
        perLayerProjectionNorm = try device.makeNormWeightTensor(archive: archive, name: "model.language_model.per_layer_projection_norm.weight")
        layers = try (0..<layerCount).map {
            try LayerWeights(device: device, archive: archive, index: $0)
        }
    }

    // MARK: Internal

    let embedTokens: Tensor
    let embedTokensPerLayer: Tensor
    let finalNorm: Tensor
    let perLayerModelProjection: Tensor
    let perLayerProjectionNorm: Tensor
    let layers: [LayerWeights]

    var allBuffers: [MTLBuffer] {
        var buffers: [MTLBuffer] = [
            embedTokens.buffer,
            embedTokensPerLayer.buffer,
            finalNorm.buffer,
            perLayerModelProjection.buffer,
            perLayerProjectionNorm.buffer
        ]
        for layer in layers {
            buffers.append(contentsOf: layer.allBuffers)
        }
        return buffers
    }
}

// MARK: - LayerWeights

struct LayerWeights {
    // MARK: Lifecycle

    init(device: MTLDevice, archive: SafeTensors, index: Int) throws {
        let prefix = "model.language_model.layers.\(index)"
        inputLayerNorm = try device.makeNormWeightTensor(archive: archive, name: "\(prefix).input_layernorm.weight")
        postAttentionLayerNorm = try device.makeNormWeightTensor(archive: archive, name: "\(prefix).post_attention_layernorm.weight")
        preFeedforwardLayerNorm = try device.makeNormWeightTensor(archive: archive, name: "\(prefix).pre_feedforward_layernorm.weight")
        postFeedforwardLayerNorm = try device.makeNormWeightTensor(archive: archive, name: "\(prefix).post_feedforward_layernorm.weight")
        postPerLayerInputNorm = try device.makeNormWeightTensor(archive: archive, name: "\(prefix).post_per_layer_input_norm.weight")
        qNorm = try device.makeNormWeightTensor(archive: archive, name: "\(prefix).self_attn.q_norm.weight")
        kNorm = try? device.makeNormWeightTensor(archive: archive, name: "\(prefix).self_attn.k_norm.weight")
        qProj = try archive.popcornTensor("\(prefix).self_attn.q_proj.weight")
        kProj = try? archive.popcornTensor("\(prefix).self_attn.k_proj.weight")
        vProj = try? archive.popcornTensor("\(prefix).self_attn.v_proj.weight")
        oProj = try archive.popcornTensor("\(prefix).self_attn.o_proj.weight")
        gateProj = try archive.popcornTensor("\(prefix).mlp.gate_proj.weight")
        upProj = try archive.popcornTensor("\(prefix).mlp.up_proj.weight")
        downProj = try archive.popcornTensor("\(prefix).mlp.down_proj.weight")
        perLayerInputGate = try archive.popcornTensor("\(prefix).per_layer_input_gate.weight")
        perLayerProjection = try archive.popcornTensor("\(prefix).per_layer_projection.weight")

        let scalar = try archive.tensor(named: "\(prefix).layer_scalar")
        layerScalar = scalar.buffer.contents().bindMemory(to: UInt16.self, capacity: 1)[0].bf16Float
        intermediateSize = gateProj.shape[0]
    }

    // MARK: Internal

    let inputLayerNorm: Tensor
    let postAttentionLayerNorm: Tensor
    let preFeedforwardLayerNorm: Tensor
    let postFeedforwardLayerNorm: Tensor
    let postPerLayerInputNorm: Tensor
    let qNorm: Tensor
    let kNorm: Tensor?
    let qProj: Tensor
    let kProj: Tensor?
    let vProj: Tensor?
    let oProj: Tensor
    let gateProj: Tensor
    let upProj: Tensor
    let downProj: Tensor
    let perLayerInputGate: Tensor
    let perLayerProjection: Tensor
    let layerScalar: Float
    let intermediateSize: Int

    var allBuffers: [MTLBuffer] {
        var buffers: [MTLBuffer] = [
            inputLayerNorm.buffer,
            postAttentionLayerNorm.buffer,
            preFeedforwardLayerNorm.buffer,
            postFeedforwardLayerNorm.buffer,
            postPerLayerInputNorm.buffer,
            qNorm.buffer,
            qProj.buffer,
            oProj.buffer,
            gateProj.buffer,
            upProj.buffer,
            downProj.buffer,
            perLayerInputGate.buffer,
            perLayerProjection.buffer
        ]
        if let kNorm { buffers.append(kNorm.buffer) }
        if let kProj { buffers.append(kProj.buffer) }
        if let vProj { buffers.append(vProj.buffer) }
        return buffers
    }
}

extension SafeTensors {
    func popcornTensor(_ name: String) throws -> Tensor {
        let safeTensor = try tensor(named: name)
        return Tensor(buffer: safeTensor.buffer, shape: safeTensor.shape, dataType: safeTensor.dtype.popcornDataType)
    }
}

extension SafeTensors.DType {
    var popcornDataType: Tensor.DataType {
        switch self {
        case .u8, .bool: .u8
        case .u16: .u16
        case .u32: .u32
        case .u64: .u64
        case .i8: .i8
        case .i16: .i16
        case .i32: .i32
        case .i64: .i64
        case .f16: .f16
        case .bf16: .bf16
        case .f32: .f32
        case .f64: .f32
        }
    }
}
