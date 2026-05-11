import Metal

@available(macOS 26.0, iOS 26.0, *) public extension MTL4ComputeCommandEncoder {
    func encode(
        _ kernel: some Kernel,
        using kernelLibrary: KernelLibrary,
        argumentTable: any MTL4ArgumentTable,
        constants: some Metal4Constants
    ) throws {
        precondition(
            kernel.tensors.count + kernel.constants.count <= 31,
            "MTL4ArgumentTable supports at most 31 buffer bindings"
        )

        let pipelineState = try kernelLibrary.pipelineState(for: kernel.functionName)
        setComputePipelineState(pipelineState)

        for (index, binding) in kernel.tensors.enumerated() {
            argumentTable.setAddress(binding.tensor.buffer.gpuAddress, index: index)
        }
        for (index, value) in kernel.constants.enumerated() {
            let address = try constants.append(value)
            argumentTable.setAddress(address, index: kernel.tensors.count + index)
        }

        setArgumentTable(argumentTable)

        let dispatchSize = kernel.dispatchSize(for: pipelineState)
        dispatchThreads(
            threadsPerGrid: dispatchSize.grid,
            threadsPerThreadgroup: dispatchSize.threadgroupSize
        )
    }
}
