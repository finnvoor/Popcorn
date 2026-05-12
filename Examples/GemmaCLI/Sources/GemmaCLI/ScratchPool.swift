@preconcurrency import Metal
import Popcorn

// MARK: - ScratchPool

final class ScratchPool: KernelScratchAllocator, @unchecked Sendable {
    // MARK: Lifecycle

    init(device: MTLDevice, residencySet: (any MTLResidencySet)? = nil, label: String = "gemma.scratch") {
        self.device = device
        self.residencySet = residencySet
        self.label = label
    }

    // MARK: Internal

    func reset() {
        lock.lock()
        free = free.map { FreeEntry(size: $0.size, buffer: $0.buffer, needsBarrier: false) }
        free.append(contentsOf: inUse.map { FreeEntry(size: $0.length, buffer: $0, needsBarrier: false) })
        inUse.removeAll(keepingCapacity: true)
        lock.unlock()
    }

    func preallocate(_ specs: [ScratchSpec]) throws {
        let buffers = try specs.map { try borrow(bytes: $0.byteCount).buffer }
        for buffer in buffers.reversed() {
            release(buffer)
        }
        reset()
    }

    func borrowTemporaryBuffer(length: Int) throws -> KernelTemporaryBuffer {
        let borrowed = try borrow(bytes: length)
        return KernelTemporaryBuffer(buffer: borrowed.buffer, wasReused: borrowed.wasReused)
    }

    func releaseTemporaryBuffer(_ temporary: KernelTemporaryBuffer) {
        release(temporary.buffer)
    }

    func borrow(bytes requestedBytes: Int) throws -> (buffer: MTLBuffer, wasReused: Bool) {
        let bytes = max(1, requestedBytes)
        lock.lock()
        defer { lock.unlock() }

        if let index = free.indices.filter({ free[$0].size >= bytes }).min(by: { free[$0].size < free[$1].size }) {
            let entry = free.remove(at: index)
            inUse.append(entry.buffer)
            return (entry.buffer, entry.needsBarrier)
        }

        guard let buffer = device.makeBuffer(length: bytes, options: [.storageModePrivate]) else {
            throw PopcornError.tensorAllocationFailed(byteCount: bytes)
        }
        buffer.label = "\(label).\(bytes)"
        residencySet?.addAllocation(buffer)
        residencySet?.commit()
        residencySet?.requestResidency()
        inUse.append(buffer)
        return (buffer, false)
    }

    func release(_ buffer: MTLBuffer) {
        lock.lock()
        if let index = inUse.firstIndex(where: { $0 === buffer }) {
            let released = inUse.remove(at: index)
            free.append(FreeEntry(size: released.length, buffer: released, needsBarrier: true))
        }
        lock.unlock()
    }

    // MARK: Private

    private struct FreeEntry {
        let size: Int
        let buffer: MTLBuffer
        let needsBarrier: Bool
    }

    private let device: MTLDevice
    private let residencySet: (any MTLResidencySet)?
    private let label: String
    private let lock = NSLock()
    private var free: [FreeEntry] = []
    private var inUse: [MTLBuffer] = []
}
