@preconcurrency import Metal
import Popcorn

// MARK: - Metal4KernelResources

@available(macOS 26.0, iOS 26.0, *) final class Metal4KernelResources: Metal4KernelResourceProvider, @unchecked Sendable {
    // MARK: Lifecycle

    init(device: MTLDevice, residencySet: (any MTLResidencySet)? = nil, constantPageSize: Int = 64 * 1024) {
        argumentTables = Metal4ArgumentTablePool(device: device)
        constants = Metal4ConstantArena(device: device, residencySet: residencySet, pageSize: constantPageSize)
        scratch = ScratchPool(device: device, residencySet: residencySet)
    }

    // MARK: Internal

    func reset() {
        argumentTables.reset()
        constants.reset()
        scratch.reset()
    }

    func preallocateConstants(pageCount: Int) throws {
        try constants.preallocate(pageCount: pageCount)
    }

    func preallocateScratch(_ specs: [ScratchSpec]) throws {
        try scratch.preallocate(specs)
    }

    func nextArgumentTable() throws -> any MTL4ArgumentTable {
        try argumentTables.next()
    }

    func appendConstant(_ bytes: UnsafeRawBufferPointer, alignment: Int) throws -> MTLGPUAddress {
        try constants.append(bytes, alignment: alignment)
    }

    func borrowTemporaryBuffer(length: Int) throws -> KernelTemporaryBuffer {
        try scratch.borrowTemporaryBuffer(length: length)
    }

    func releaseTemporaryBuffer(_ temporary: KernelTemporaryBuffer) {
        scratch.releaseTemporaryBuffer(temporary)
    }

    // MARK: Private

    private let argumentTables: Metal4ArgumentTablePool
    private let constants: Metal4ConstantArena
    private let scratch: ScratchPool
}
