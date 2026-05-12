@preconcurrency import Metal
import Popcorn

// MARK: - Metal4KernelContext

final class Metal4KernelContext: KernelContext {
    // MARK: Lifecycle

    init(
        device: MTLDevice,
        residencyLabel: String,
        residencyCapacity: Int,
        constantPageSize: Int
    ) throws {
        self.device = device
        kernelLibrary = try KernelLibrary(device: device)

        guard let commandQueue = device.makeMTL4CommandQueue() else {
            throw GemmaError.message("Could not create MTL4CommandQueue.")
        }
        self.commandQueue = commandQueue

        guard let commandAllocator = device.makeCommandAllocator() else {
            throw GemmaError.message("Could not create MTL4CommandAllocator.")
        }
        self.commandAllocator = commandAllocator

        guard let event = device.makeEvent() else {
            throw GemmaError.message("Could not create MTLEvent for queue serialization.")
        }
        serialEvent = event

        let residencyDescriptor = MTLResidencySetDescriptor()
        residencyDescriptor.label = residencyLabel
        residencyDescriptor.initialCapacity = residencyCapacity
        residencySet = try device.makeResidencySet(descriptor: residencyDescriptor)
        resources = Metal4KernelResources(
            device: device,
            residencySet: residencySet,
            constantPageSize: constantPageSize
        )
        try resources.preallocateConstants(pageCount: 1)
    }

    // MARK: Internal

    let resources: Metal4KernelResources
    var profiler: Metal4DispatchProfiler?

    func preallocateScratch(_ specs: [ScratchSpec]) throws {
        try resources.preallocateScratch(specs)
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
        guard let commandBuffer = device.makeCommandBuffer() else {
            throw GemmaError.message("Could not create MTL4CommandBuffer.")
        }
        commandBuffer.beginCommandBuffer(allocator: commandAllocator)

        guard let compute = commandBuffer.makeComputeCommandEncoder() else {
            throw GemmaError.message("Could not create MTL4ComputeCommandEncoder.")
        }

        stateLock.lock()
        let cbIndex = totalSubmits
        totalSubmits += 1
        stateLock.unlock()
        commandBuffer.label = "gemma4.cb.\(cbIndex)"
        compute.label = "gemma4.compute.\(cbIndex)"

        let profilerSnapshot: Metal4DispatchProfiler.ResolvedSnapshot?
        if let profiler {
            profiler.beginCommandBuffer(index: cbIndex)
            let kernelEncoder = try KernelCommandEncoder(
                compute: compute,
                library: kernelLibrary,
                resources: resources
            )
            kernelEncoder.metal4DispatchHook = { name, phase, encoder in
                profiler.dispatchHook(kernelName: name, phase: phase, encoder: encoder)
            }
            try encode(kernelEncoder)
            profilerSnapshot = profiler.endCommandBuffer()
        } else {
            let kernelEncoder = try KernelCommandEncoder(
                compute: compute,
                library: kernelLibrary,
                resources: resources
            )
            try encode(kernelEncoder)
            profilerSnapshot = nil
        }

        compute.endEncoding()
        commandBuffer.endCommandBuffer()

        // The MTL4 queue doesn't auto-serialize CBs that share buffers (workspace state,
        // constant arena pages, scratch buffers). Chain CBs with an MTLEvent so each CB's
        // GPU work waits for the previous CB to complete. CPU encoding still overlaps with
        // GPU execution, which is what gives pipelined decoding its win.
        stateLock.lock()
        serialEventValue += 1
        let signalValue = serialEventValue
        let waitValue = signalValue - 1
        inFlightCount += 1
        stateLock.unlock()

        if waitValue > 0 {
            commandQueue.waitForEvent(serialEvent, value: waitValue)
        }

        let feedback = CommitFeedbackBox()
        let options = MTL4CommitOptions()
        options.addFeedbackHandler { [self, profiler] commitFeedback in
            stateLock.lock()
            inFlightCount -= 1
            let drained = inFlightCount == 0
            stateLock.unlock()
            // Only reset the constant arena / scratch when the queue is fully drained.
            // Sibling CBs that are encoded but not yet executed reference offsets in
            // the arena; resetting would let later encodes clobber those bytes.
            if drained {
                resources.reset()
            }
            if let snapshot = profilerSnapshot {
                profiler?.flush(snapshot)
            }
            feedback.finish(error: commitFeedback.error)
        }
        commandQueue.commit([commandBuffer], options: options)
        commandQueue.signalEvent(serialEvent, value: signalValue)
        return feedback
    }

    // MARK: Private

    private let device: MTLDevice
    private let kernelLibrary: KernelLibrary
    private let commandQueue: any MTL4CommandQueue
    private let commandAllocator: any MTL4CommandAllocator
    private let residencySet: any MTLResidencySet
    private let serialEvent: any MTLEvent
    private let stateLock = NSLock()
    private var serialEventValue: UInt64 = 0
    private var inFlightCount: Int = 0
    private var totalSubmits: Int = 0
}
