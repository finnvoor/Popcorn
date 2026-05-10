import Foundation

// MARK: - Gemma4Config

struct Gemma4Config {
    enum LayerType: String {
        case sliding = "sliding_attention"
        case full = "full_attention"
    }

    let hiddenSize: Int
    let hiddenSizePerLayerInput: Int
    let intermediateSize: Int
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let numKVSharedLayers: Int
    let headDim: Int
    let globalHeadDim: Int
    let vocabSize: Int
    let maxPositionEmbeddings: Int
    let slidingWindow: Int
    let rmsNormEps: Float
    let finalLogitSoftcap: Float
    let eosTokenId: Int
    let layerTypes: [LayerType]

    static func load(from url: URL) throws -> Gemma4Config {
        let data = try Data(contentsOf: url)
        guard
            let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let text = root["text_config"] as? [String: Any]
        else {
            throw GemmaError.message("config.json does not contain text_config.")
        }

        let layerTypeStrings: [String] = try requireArray(text, "layer_types")
        return try Gemma4Config(
            hiddenSize: requireInt(text, "hidden_size"),
            hiddenSizePerLayerInput: requireInt(text, "hidden_size_per_layer_input"),
            intermediateSize: requireInt(text, "intermediate_size"),
            numHiddenLayers: requireInt(text, "num_hidden_layers"),
            numAttentionHeads: requireInt(text, "num_attention_heads"),
            numKeyValueHeads: requireInt(text, "num_key_value_heads"),
            numKVSharedLayers: requireInt(text, "num_kv_shared_layers"),
            headDim: requireInt(text, "head_dim"),
            globalHeadDim: requireInt(text, "global_head_dim"),
            vocabSize: requireInt(text, "vocab_size"),
            maxPositionEmbeddings: requireInt(text, "max_position_embeddings"),
            slidingWindow: requireInt(text, "sliding_window"),
            rmsNormEps: requireFloat(text, "rms_norm_eps"),
            finalLogitSoftcap: requireFloat(text, "final_logit_softcapping"),
            eosTokenId: requireInt(text, "eos_token_id"),
            layerTypes: layerTypeStrings.map { value in
                guard let layerType = LayerType(rawValue: value) else {
                    throw GemmaError.message("Unsupported layer type \(value).")
                }
                return layerType
            }
        )
    }

    func kvSourceLayer(for layerIndex: Int) -> Int {
        let firstShared = numHiddenLayers - numKVSharedLayers
        guard layerIndex >= firstShared else { return layerIndex }
        let wanted = layerTypes[layerIndex]
        for i in stride(from: firstShared - 1, through: 0, by: -1) where layerTypes[i] == wanted {
            return i
        }
        return layerIndex
    }
}

private func requireInt(_ dictionary: [String: Any], _ key: String) throws -> Int {
    guard let value = dictionary[key] as? NSNumber else {
        throw GemmaError.message("Missing integer config key \(key).")
    }
    return value.intValue
}

private func requireFloat(_ dictionary: [String: Any], _ key: String) throws -> Float {
    guard let value = dictionary[key] as? NSNumber else {
        throw GemmaError.message("Missing float config key \(key).")
    }
    return value.floatValue
}

private func requireArray<T>(_ dictionary: [String: Any], _ key: String) throws -> [T] {
    guard let value = dictionary[key] as? [T] else {
        throw GemmaError.message("Missing array config key \(key).")
    }
    return value
}
