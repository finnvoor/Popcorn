@preconcurrency import Metal
import Popcorn

// MARK: - MetalKernelContext

final class MetalKernelContext: KernelContext {
    // MARK: Lifecycle

    init(
        device: MTLDevice,
        residencyLabel: String,
        residencyCapacity: Int
    ) throws {
        self.device = device
        kernelLibrary = try KernelLibrary(device: device)

        guard let commandQueue = device.makeCommandQueue() else {
            throw GemmaError.message("Could not create MTLCommandQueue.")
        }
        self.commandQueue = commandQueue

        let residencyDescriptor = MTLResidencySetDescriptor()
        residencyDescriptor.label = residencyLabel
        residencyDescriptor.initialCapacity = residencyCapacity
        residencySet = try device.makeResidencySet(descriptor: residencyDescriptor)
        scratch = ScratchPool(device: device, residencySet: residencySet)
    }

    // MARK: Internal

    func preallocateScratch(_ specs: [ScratchSpec]) throws {
        try scratch.preallocate(specs)
    }

    func addResidency(_ buffers: [MTLBuffer]) {
        for buffer in buffers {
            residencySet.addAllocation(buffer)
        }
    }

    func commitResidency() {
        residencySet.commit()
        residencySet.requestResidency()
        commandQueue.addResidencySet(residencySet)
    }

    func submit(_ encode: (KernelCommandEncoder) throws -> Void) throws -> CommitFeedbackBox {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GemmaError.message("Could not create MTLCommandBuffer.")
        }
        guard let compute = commandBuffer.makeComputeCommandEncoder(dispatchType: .concurrent) else {
            throw GemmaError.message("Could not create MTLComputeCommandEncoder.")
        }

        let cbIndex = totalSubmits
        totalSubmits += 1
        commandBuffer.label = "gemma4.cb.\(cbIndex)"
        compute.label = "gemma4.compute.\(cbIndex)"

        let kernelEncoder = KernelCommandEncoder(
            compute: compute,
            library: kernelLibrary,
            scratch: scratch
        )
        try encode(kernelEncoder)
        compute.endEncoding()

        let feedback = CommitFeedbackBox()
        commandBuffer.addCompletedHandler { [scratch] completed in
            scratch.reset()
            feedback.finish(error: completed.error)
        }
        commandBuffer.commit()
        return feedback
    }

    // MARK: Private

    private let device: MTLDevice
    private let kernelLibrary: KernelLibrary
    private let commandQueue: any MTLCommandQueue
    private let residencySet: any MTLResidencySet
    private let scratch: ScratchPool
    private var totalSubmits: Int = 0
}
