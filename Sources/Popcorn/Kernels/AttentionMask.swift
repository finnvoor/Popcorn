import Foundation

// MARK: - AttentionMask

/// Which keys a query can attend to in `Kernels.FlashAttention`.
///
/// The three cases cover the full set of masks any current Popcorn attention
/// workload needs:
///
/// - `.causal` is the standard autoregressive LLM mask: a query at position
///   `q` may attend to keys at positions `0...q`.
/// - `.causalSlidingWindow(_:)` further restricts that to the most recent
///   `window` keys (Gemma-style local layers).
/// - `.bidirectional` removes the mask entirely. Every query attends to
///   every key; this is what encoder-only audio/text models
///   (Wav2Vec2 / HuBERT, BERT-family) want.
public enum AttentionMask: Sendable, Equatable {
    case causal
    case causalSlidingWindow(window: Int)
    case bidirectional
}

extension AttentionMask {
    /// Tag the kernel constants struct uses to switch on the mask.
    ///   0 = causal, 1 = causal + sliding window, 2 = bidirectional.
    var kindRawValue: UInt32 {
        switch self {
        case .causal: 0
        case .causalSlidingWindow: 1
        case .bidirectional: 2
        }
    }

    /// Sliding-window length for `.causalSlidingWindow`. Unused otherwise;
    /// passed as `-1` so the kernel's `>= 0` guard naturally ignores it.
    var slidingWindow: Int32 {
        if case let .causalSlidingWindow(window) = self {
            return Int32(window)
        }
        return -1
    }
}
