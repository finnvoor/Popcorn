import Metal

// MARK: - Kernels

public enum Kernels {}

// MARK: - Kernel

public protocol Kernel {
    var functionName: String { get }

    var tensors: [Tensor.Binding] { get }
    var constants: [any BitwiseCopyable] { get }

    var grid: MTLSize { get }
    var threadgroupSize: MTLSize { get }
}

public extension Kernel {
    var buffers: [MTLBuffer] {
        tensors.map(\.tensor.buffer)
    }

    var constants: [any BitwiseCopyable] {
        []
    }
}
