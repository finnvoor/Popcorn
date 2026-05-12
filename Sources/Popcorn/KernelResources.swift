import Metal

// MARK: - KernelTemporaryBuffer

public struct KernelTemporaryBuffer {
    // MARK: Lifecycle

    public init(buffer: MTLBuffer, wasReused: Bool = false) {
        self.buffer = buffer
        self.wasReused = wasReused
    }

    // MARK: Public

    public let buffer: MTLBuffer

    /// True when this buffer may contain writes from earlier GPU work in the same
    /// command encoder. Popcorn will conservatively treat it as previously written
    /// so ordinary hazard tracking inserts a barrier before reading it.
    public let wasReused: Bool
}

// MARK: - KernelScratchAllocator

/// Scratch-buffer provider for composite kernels.
///
/// Simple kernels do not use scratch, but every encoder has an allocator so
/// temporary-buffer ownership is always explicit.
public protocol KernelScratchAllocator: AnyObject {
    func borrowTemporaryBuffer(length: Int) throws -> KernelTemporaryBuffer
    func releaseTemporaryBuffer(_ temporary: KernelTemporaryBuffer)
}

public extension KernelScratchAllocator {
    func releaseTemporaryBuffer(_: KernelTemporaryBuffer) {}
}

// MARK: - Metal4KernelResourceProvider

/// Resource provider for Popcorn's Metal 4 encoder.
///
/// Popcorn performs the dispatch: it sets the PSO, binds all buffers and
/// constants into the supplied argument table, computes grid/threadgroup sizes,
/// emits barriers when hazard tracking requires them, and calls dispatchThreads.
/// The provider only supplies reusable resources owned by the caller's runtime.
@available(macOS 26.0, iOS 26.0, *) public protocol Metal4KernelResourceProvider: KernelScratchAllocator {
    func nextArgumentTable() throws -> any MTL4ArgumentTable
    func appendConstant(_ bytes: UnsafeRawBufferPointer, alignment: Int) throws -> MTLGPUAddress
}

@available(macOS 26.0, iOS 26.0, *) public extension Metal4KernelResourceProvider {
    func appendConstant<T: BitwiseCopyable>(_ value: T) throws -> MTLGPUAddress {
        try withUnsafeBytes(of: value) { bytes in
            try appendConstant(bytes, alignment: MemoryLayout<T>.alignment)
        }
    }
}
