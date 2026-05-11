import Metal

// MARK: - Kernels

public enum Kernels {}

// MARK: - Kernel

public protocol Kernel {
    var functionName: String { get }

    var tensors: [Tensor.Binding] { get }
    var constants: [any BitwiseCopyable] { get }

    func dispatchSize(for pipelineState: MTLComputePipelineState) -> (grid: MTLSize, threadgroupSize: MTLSize)
}

public extension Kernel {
    var buffers: [MTLBuffer] {
        tensors.map(\.tensor.buffer)
    }

    var constants: [any BitwiseCopyable] {
        []
    }
}
