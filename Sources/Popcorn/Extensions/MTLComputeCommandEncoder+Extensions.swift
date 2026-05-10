import Metal

public extension MTLComputeCommandEncoder {
    func encode(_ kernel: some Kernel, using kernelLibrary: KernelLibrary) throws {
        let pipelineState = try kernelLibrary.pipelineState(for: kernel.functionName)
        setComputePipelineState(pipelineState)

        for (index, binding) in kernel.tensors.enumerated() {
            setBuffer(binding.tensor.buffer, offset: 0, index: index)
        }
        for (index, value) in kernel.constants.enumerated() {
            setConstants(value, index: kernel.tensors.count + index)
        }

        dispatchThreads(kernel.grid, threadsPerThreadgroup: kernel.threadgroupSize)
    }

    /// necessary to open the existential
    @inlinable func setConstants<T: BitwiseCopyable>(_ value: T, index: Int) {
        var v = value
        setBytes(&v, length: MemoryLayout<T>.stride, index: index)
    }
}
