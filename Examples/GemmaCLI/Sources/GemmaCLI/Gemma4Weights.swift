import Metal
import MTLSafeTensors
import Popcorn

// MARK: - Gemma4Weights

struct Gemma4Weights {
    // MARK: Lifecycle

    init(
        device: MTLDevice,
        archive: SafeTensors,
        layerCount: Int,
        quantization: Gemma4Config.Quantization?
    ) throws {
        let loader = try WeightLoader(archive: archive, quantization: quantization)
        embedTokens = try loader.embedding(loader.qualify("embed_tokens"))
        embedTokensPerLayer = try loader.embedding(loader.qualify("embed_tokens_per_layer"))
        finalNorm = try device.makeNormWeightTensor(archive: archive, name: loader.qualify("norm.weight"))
        perLayerModelProjection = try loader.linear(loader.qualify("per_layer_model_projection"))
        perLayerProjectionNorm = try device.makeNormWeightTensor(
            archive: archive,
            name: loader.qualify("per_layer_projection_norm.weight")
        )
        layers = try (0..<layerCount).map {
            try LayerWeights(device: device, loader: loader, index: $0)
        }
    }

    // MARK: Internal

    let embedTokens: EmbeddingTable
    let embedTokensPerLayer: EmbeddingTable
    let finalNorm: Tensor
    let perLayerModelProjection: LinearWeight
    let perLayerProjectionNorm: Tensor
    let layers: [LayerWeights]

    var allBuffers: [MTLBuffer] {
        var buffers: [MTLBuffer] = []
        buffers.append(contentsOf: embedTokens.allBuffers)
        buffers.append(contentsOf: embedTokensPerLayer.allBuffers)
        buffers.append(finalNorm.buffer)
        buffers.append(contentsOf: perLayerModelProjection.allBuffers)
        buffers.append(perLayerProjectionNorm.buffer)
        for layer in layers {
            buffers.append(contentsOf: layer.allBuffers)
        }
        return buffers
    }
}

// MARK: - LayerWeights

struct LayerWeights {
    // MARK: Lifecycle

    init(device: MTLDevice, loader: WeightLoader, index: Int) throws {
        let layerPrefix = loader.layerPrefix(index)
        inputLayerNorm = try device.makeNormWeightTensor(archive: loader.archive, name: "\(layerPrefix).input_layernorm.weight")
        postAttentionLayerNorm = try device.makeNormWeightTensor(archive: loader.archive, name: "\(layerPrefix).post_attention_layernorm.weight")
        preFeedforwardLayerNorm = try device.makeNormWeightTensor(archive: loader.archive, name: "\(layerPrefix).pre_feedforward_layernorm.weight")
        postFeedforwardLayerNorm = try device.makeNormWeightTensor(archive: loader.archive, name: "\(layerPrefix).post_feedforward_layernorm.weight")
        postPerLayerInputNorm = try device.makeNormWeightTensor(archive: loader.archive, name: "\(layerPrefix).post_per_layer_input_norm.weight")
        qNorm = try device.makeNormWeightTensor(archive: loader.archive, name: "\(layerPrefix).self_attn.q_norm.weight")
        kNorm = try? device.makeNormWeightTensor(archive: loader.archive, name: "\(layerPrefix).self_attn.k_norm.weight")
        qProj = try loader.linear("\(layerPrefix).self_attn.q_proj")
        kProj = try? loader.linear("\(layerPrefix).self_attn.k_proj")
        vProj = try? loader.linear("\(layerPrefix).self_attn.v_proj")
        oProj = try loader.linear("\(layerPrefix).self_attn.o_proj")
        gateProj = try loader.linear("\(layerPrefix).mlp.gate_proj")
        upProj = try loader.linear("\(layerPrefix).mlp.up_proj")
        downProj = try loader.linear("\(layerPrefix).mlp.down_proj")
        perLayerInputGate = try loader.linear("\(layerPrefix).per_layer_input_gate")
        perLayerProjection = try loader.linear("\(layerPrefix).per_layer_projection")

        let scalar = try loader.archive.tensor(named: "\(layerPrefix).layer_scalar")
        layerScalar = scalar.buffer.contents().bindMemory(to: UInt16.self, capacity: 1)[0].bf16Float
        intermediateSize = gateProj.outFeatures
    }

    // MARK: Internal

    let inputLayerNorm: Tensor
    let postAttentionLayerNorm: Tensor
    let preFeedforwardLayerNorm: Tensor
    let postFeedforwardLayerNorm: Tensor
    let postPerLayerInputNorm: Tensor
    let qNorm: Tensor
    let kNorm: Tensor?
    let qProj: LinearWeight
    let kProj: LinearWeight?
    let vProj: LinearWeight?
    let oProj: LinearWeight
    let gateProj: LinearWeight
    let upProj: LinearWeight
    let downProj: LinearWeight
    let perLayerInputGate: LinearWeight
    let perLayerProjection: LinearWeight
    let layerScalar: Float
    let intermediateSize: Int

    var allBuffers: [MTLBuffer] {
        var buffers: [MTLBuffer] = [
            inputLayerNorm.buffer,
            postAttentionLayerNorm.buffer,
            preFeedforwardLayerNorm.buffer,
            postFeedforwardLayerNorm.buffer,
            postPerLayerInputNorm.buffer,
            qNorm.buffer
        ]
        buffers.append(contentsOf: qProj.allBuffers)
        buffers.append(contentsOf: oProj.allBuffers)
        buffers.append(contentsOf: gateProj.allBuffers)
        buffers.append(contentsOf: upProj.allBuffers)
        buffers.append(contentsOf: downProj.allBuffers)
        buffers.append(contentsOf: perLayerInputGate.allBuffers)
        buffers.append(contentsOf: perLayerProjection.allBuffers)
        if let kNorm { buffers.append(kNorm.buffer) }
        if let kProj { buffers.append(contentsOf: kProj.allBuffers) }
        if let vProj { buffers.append(contentsOf: vProj.allBuffers) }
        return buffers
    }
}
