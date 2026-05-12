import Metal

// MARK: - KernelCommandEncoder

public final class KernelCommandEncoder {
    // MARK: Lifecycle

    public init(
        compute: any MTLComputeCommandEncoder,
        library: KernelLibrary,
        scratch: any KernelScratchAllocator
    ) {
        backend = .metal(compute, scratch)
        self.library = library
        device = compute.device
    }

    @available(macOS 26.0, iOS 26.0, *) public init(
        compute: any MTL4ComputeCommandEncoder,
        library: KernelLibrary,
        resources: any Metal4KernelResourceProvider
    ) throws {
        backend = try .metal4(compute, resources, resources.nextArgumentTable())
        self.library = library
        device = library.device
    }

    // MARK: Public

    public let device: MTLDevice

    /// Optional per-dispatch hook for the Metal 4 backend. Useful for timestamp
    /// profiling. Has no effect on the Metal backend.
    @available(macOS 26.0, iOS 26.0, *) public var metal4DispatchHook: Metal4DispatchHook? {
        get { _metal4DispatchHook as? Metal4DispatchHook }
        set { _metal4DispatchHook = newValue }
    }

    public func encode(_ kernel: some Kernel) throws {
        try kernel.encode(to: self)
    }

    public func encode(@KernelBuilder _ body: () throws -> [any Kernel]) throws {
        for kernel in try body() {
            try kernel.encode(to: self)
        }
    }

    public func dispatch(_ kernel: some DispatchKernel) throws {
        if needsBarrier(for: kernel) {
            barrier()
        }

        switch backend {
        case let .metal(compute, _):
            try encodeMetal(kernel, on: compute)
        case let .metal4(compute, resources, argumentTable):
            if #available(macOS 26.0, iOS 26.0, *) {
                try encodeMetal4(
                    kernel,
                    on: compute as! any MTL4ComputeCommandEncoder,
                    resources: resources as! any Metal4KernelResourceProvider,
                    argumentTable: argumentTable as! any MTL4ArgumentTable
                )
            }
        }

        recordAccesses(kernel.tensors)
    }

    public func barrier() {
        switch backend {
        case let .metal(compute, _):
            compute.memoryBarrier(scope: .buffers)
        case let .metal4(compute, _, _):
            if #available(macOS 26.0, iOS 26.0, *) {
                (compute as! any MTL4ComputeCommandEncoder).barrier(
                    afterEncoderStages: .dispatch,
                    beforeEncoderStages: .dispatch,
                    visibilityOptions: .device
                )
            }
        }
        accessByBuffer.removeAll(keepingCapacity: true)
    }

    public func withTemporaryTensor<R>(
        _ spec: ScratchSpec,
        _ body: (Tensor) throws -> R
    ) throws -> R {
        let temporary = try borrowTemporaryBuffer(length: spec.byteCount)
        if temporary.wasReused { accessByBuffer[temporary.buffer.gpuAddress] = .write }
        let tensor = Tensor(buffer: temporary.buffer, shape: spec.shape, dataType: spec.dataType)
        defer { releaseTemporaryBuffer(temporary) }
        return try body(tensor)
    }

    public func withTemporaryTensor<R>(
        _ s1: ScratchSpec,
        _ s2: ScratchSpec,
        _ body: (Tensor, Tensor) throws -> R
    ) throws -> R {
        try withTemporaryTensor(s1) { t1 in
            try withTemporaryTensor(s2) { t2 in
                try body(t1, t2)
            }
        }
    }

    public func withTemporaryTensor<R>(
        _ s1: ScratchSpec,
        _ s2: ScratchSpec,
        _ s3: ScratchSpec,
        _ body: (Tensor, Tensor, Tensor) throws -> R
    ) throws -> R {
        try withTemporaryTensor(s1) { t1 in
            try withTemporaryTensor(s2) { t2 in
                try withTemporaryTensor(s3) { t3 in
                    try body(t1, t2, t3)
                }
            }
        }
    }

    public func withTemporaryTensor<R>(
        _ s1: ScratchSpec,
        _ s2: ScratchSpec,
        _ s3: ScratchSpec,
        _ s4: ScratchSpec,
        _ body: (Tensor, Tensor, Tensor, Tensor) throws -> R
    ) throws -> R {
        try withTemporaryTensor(s1) { t1 in
            try withTemporaryTensor(s2) { t2 in
                try withTemporaryTensor(s3) { t3 in
                    try withTemporaryTensor(s4) { t4 in
                        try body(t1, t2, t3, t4)
                    }
                }
            }
        }
    }

    // MARK: Private

    private enum Backend {
        case metal(any MTLComputeCommandEncoder, any KernelScratchAllocator)
        case metal4(Any, Any, Any)
    }

    private let backend: Backend
    private let library: KernelLibrary
    private var accessByBuffer: [MTLGPUAddress: Tensor.Access] = [:]
    private var _metal4DispatchHook: Any?

    private func needsBarrier(for kernel: some DispatchKernel) -> Bool {
        for binding in kernel.tensors {
            guard let previous = accessByBuffer[binding.tensor.buffer.gpuAddress] else { continue }
            if previous.contains(.write), !binding.access.isDisjoint(with: [.read, .write]) { return true }
            if binding.access.contains(.write), !previous.isDisjoint(with: [.read, .write]) { return true }
        }
        return false
    }

    private func recordAccesses(_ bindings: [Tensor.Binding]) {
        for binding in bindings {
            accessByBuffer[binding.tensor.buffer.gpuAddress, default: []].formUnion(binding.access)
        }
    }

    private func borrowTemporaryBuffer(length: Int) throws -> KernelTemporaryBuffer {
        switch backend {
        case let .metal(_, scratch):
            return try scratch.borrowTemporaryBuffer(length: length)
        case let .metal4(_, resources, _):
            if #available(macOS 26.0, iOS 26.0, *) {
                return try (resources as! any Metal4KernelResourceProvider).borrowTemporaryBuffer(length: length)
            }
            throw PopcornError.kernelFunctionNotFound("metal4")
        }
    }

    private func releaseTemporaryBuffer(_ temporary: KernelTemporaryBuffer) {
        switch backend {
        case let .metal(_, scratch):
            scratch.releaseTemporaryBuffer(temporary)
        case let .metal4(_, resources, _):
            if #available(macOS 26.0, iOS 26.0, *) {
                (resources as! any Metal4KernelResourceProvider).releaseTemporaryBuffer(temporary)
            }
        }
    }

    private func encodeMetal(_ kernel: some DispatchKernel, on compute: any MTLComputeCommandEncoder) throws {
        let pipelineState = try library.pipelineState(for: kernel.functionName)
        compute.pushDebugGroup(kernel.functionName)
        defer { compute.popDebugGroup() }
        compute.setComputePipelineState(pipelineState)

        for (index, binding) in kernel.tensors.enumerated() {
            compute.setBuffer(binding.tensor.buffer, offset: 0, index: index)
        }
        for (index, value) in kernel.constants.enumerated() {
            compute.setConstants(value, index: kernel.tensors.count + index)
        }

        let dispatchSize = kernel.dispatchSize(for: pipelineState)
        compute.dispatchThreads(dispatchSize.grid, threadsPerThreadgroup: dispatchSize.threadgroupSize)
    }

    @available(macOS 26.0, iOS 26.0, *) private func encodeMetal4(
        _ kernel: some DispatchKernel,
        on compute: any MTL4ComputeCommandEncoder,
        resources: any Metal4KernelResourceProvider,
        argumentTable: any MTL4ArgumentTable
    ) throws {
        precondition(
            kernel.tensors.count + kernel.constants.count <= 31,
            "MTL4ArgumentTable supports at most 31 buffer bindings"
        )

        let pipelineState = try library.pipelineState(for: kernel.functionName)
        compute.pushDebugGroup(kernel.functionName)
        defer { compute.popDebugGroup() }
        compute.setComputePipelineState(pipelineState)

        for (index, binding) in kernel.tensors.enumerated() {
            argumentTable.setAddress(binding.tensor.buffer.gpuAddress, index: index)
        }
        for (index, value) in kernel.constants.enumerated() {
            let address = try resources.appendConstant(value)
            argumentTable.setAddress(address, index: kernel.tensors.count + index)
        }

        compute.setArgumentTable(argumentTable)

        let dispatchSize = kernel.dispatchSize(for: pipelineState)
        metal4DispatchHook?(kernel.functionName, .willDispatch, compute)
        compute.dispatchThreads(threadsPerGrid: dispatchSize.grid, threadsPerThreadgroup: dispatchSize.threadgroupSize)
        metal4DispatchHook?(kernel.functionName, .didDispatch, compute)
    }
}

// MARK: - Metal4DispatchPhase

@available(macOS 26.0, iOS 26.0, *) public enum Metal4DispatchPhase {
    case willDispatch
    case didDispatch
}

@available(macOS 26.0, iOS 26.0, *) public typealias Metal4DispatchHook = (_ kernelName: String, _ phase: Metal4DispatchPhase, _ encoder: any MTL4ComputeCommandEncoder) -> Void

private extension MTLComputeCommandEncoder {
    /// Necessary to open the existential.
    func setConstants<T: BitwiseCopyable>(_ value: T, index: Int) {
        var v = value
        setBytes(&v, length: MemoryLayout<T>.stride, index: index)
    }
}
