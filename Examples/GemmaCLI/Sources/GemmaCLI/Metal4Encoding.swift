import Dispatch
import Foundation
import Metal
import Popcorn

// MARK: - ArgumentTablePool

final class ArgumentTablePool {
    // MARK: Lifecycle

    init(device: MTLDevice) {
        self.device = device
        descriptor.maxBufferBindCount = 31
        descriptor.maxTextureBindCount = 0
        descriptor.maxSamplerStateBindCount = 0
        descriptor.initializeBindings = false
        descriptor.supportAttributeStrides = false
    }

    // MARK: Internal

    func reset() {
        index = 0
    }

    func next() throws -> any MTL4ArgumentTable {
        if index == tables.count {
            try tables.append(device.makeArgumentTable(descriptor: descriptor))
        }
        defer { index += 1 }
        return tables[index]
    }

    // MARK: Private

    private let device: MTLDevice
    private let descriptor = MTL4ArgumentTableDescriptor()
    private var tables: [any MTL4ArgumentTable] = []
    private var index = 0
}

// MARK: - HazardTrackingEncoder

final class HazardTrackingEncoder {
    // MARK: Lifecycle

    init(
        encoder: any MTL4ComputeCommandEncoder,
        kernelLibrary: KernelLibrary,
        argumentTables: ArgumentTablePool,
        constants: Metal4ConstantArena
    ) {
        self.encoder = encoder
        self.kernelLibrary = kernelLibrary
        self.argumentTables = argumentTables
        self.constants = constants
    }

    // MARK: Internal

    func encode(_ kernel: some Kernel) throws {
        if needsBarrier(for: kernel) {
            encoder.barrier(
                afterEncoderStages: .dispatch,
                beforeEncoderStages: .dispatch,
                visibilityOptions: .device
            )
            accessByBuffer.removeAll(keepingCapacity: true)
        }
        let argumentTable = try argumentTables.next()
        try encoder.encode(kernel, using: kernelLibrary, argumentTable: argumentTable, constants: constants)
        for binding in kernel.tensors {
            let key = binding.tensor.buffer.gpuAddress
            accessByBuffer[key, default: []].formUnion(binding.access)
        }
    }

    // MARK: Private

    private let encoder: any MTL4ComputeCommandEncoder
    private let kernelLibrary: KernelLibrary
    private let argumentTables: ArgumentTablePool
    private let constants: Metal4ConstantArena
    private var accessByBuffer: [MTLGPUAddress: Tensor.Access] = [:]

    private func needsBarrier(for kernel: some Kernel) -> Bool {
        for binding in kernel.tensors {
            guard let previous = accessByBuffer[binding.tensor.buffer.gpuAddress] else { continue }

            if previous.contains(.write), !binding.access.isDisjoint(with: [.read, .write]) { return true }

            if binding.access.contains(.write), !previous.isDisjoint(with: [.read, .write]) { return true }
        }
        return false
    }
}

// MARK: - CommitFeedbackBox

final class CommitFeedbackBox: @unchecked Sendable {
    // MARK: Internal

    func finish(error: (any Error)?) {
        lock.lock()
        self.error = error
        lock.unlock()
        semaphore.signal()
    }

    func wait() throws {
        semaphore.wait()
        lock.lock()
        let error = error
        lock.unlock()
        if let error { throw error }
    }

    // MARK: Private

    private let semaphore = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private var error: (any Error)?
}
