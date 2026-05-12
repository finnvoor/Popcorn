import Metal

// MARK: - Kernels

public enum Kernels {}

// MARK: - Kernel

/// A value-typed Popcorn kernel operation.
///
/// Kernels validate tensor shapes at initialization time and choose the concrete
/// Metal function, constants, threadgroup size, and multi-dispatch sequence when
/// encoded. Popcorn owns dispatch sequencing and hazard tracking; callers provide
/// scratch/Metal 4 resource pools.
public protocol Kernel {
    /// The user-visible tensor resources touched by this operation. Composite
    /// kernels expose their external resources here; temporary scratch resources
    /// are requested from the encoder while encoding.
    var tensors: [Tensor.Binding] { get }

    func encode(to encoder: KernelCommandEncoder) throws
}

// MARK: - KernelBuilder

@resultBuilder public enum KernelBuilder {
    public static func buildBlock(_ components: [any Kernel]...) -> [any Kernel] {
        components.flatMap(\.self)
    }

    public static func buildExpression(_ kernel: some Kernel) -> [any Kernel] {
        [kernel]
    }

    public static func buildOptional(_ component: [any Kernel]?) -> [any Kernel] {
        component ?? []
    }

    public static func buildEither(first component: [any Kernel]) -> [any Kernel] {
        component
    }

    public static func buildEither(second component: [any Kernel]) -> [any Kernel] {
        component
    }

    public static func buildArray(_ components: [[any Kernel]]) -> [any Kernel] {
        components.flatMap(\.self)
    }
}

// MARK: - DispatchKernel

public protocol DispatchKernel: Kernel {
    var functionName: String { get }

    var constants: [any BitwiseCopyable] { get }

    func dispatchSize(for pipelineState: MTLComputePipelineState) -> (grid: MTLSize, threadgroupSize: MTLSize)
}

public extension DispatchKernel {
    var constants: [any BitwiseCopyable] {
        []
    }

    func encode(to encoder: KernelCommandEncoder) throws {
        try encoder.dispatch(self)
    }
}
