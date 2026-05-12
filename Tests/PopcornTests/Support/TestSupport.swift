import Dispatch
import Metal
@testable import Popcorn
import Testing

// MARK: - TestSupportError

enum TestSupportError: Error {
    case bufferAllocationFailed
}

// MARK: - TestScratchAllocator

final class TestScratchAllocator: KernelScratchAllocator {
    // MARK: Lifecycle

    init(device: MTLDevice) {
        self.device = device
    }

    // MARK: Internal

    func borrowTemporaryBuffer(length: Int) throws -> KernelTemporaryBuffer {
        guard let buffer = device.makeBuffer(length: max(1, length), options: [.storageModePrivate]) else {
            throw PopcornError.tensorAllocationFailed(byteCount: length)
        }
        return KernelTemporaryBuffer(buffer: buffer)
    }

    // MARK: Private

    private let device: MTLDevice
}

// MARK: - Inputs

let inputA: [Float] = [0, 1, -2.5, 3.25, 100, -8, 0.5, 42]
let inputB: [Float] = [4, -1, 2.5, 0.75, -50, -12, 1.5, -40]

// MARK: - Buffers

func makeBuffers(device: MTLDevice) throws -> (a: Tensor, b: Tensor, out: Tensor) {
    let byteCount = inputA.count * MemoryLayout<Float>.stride

    guard let inA = device.makeBuffer(bytes: inputA, length: byteCount, options: .storageModeShared),
          let inB = device.makeBuffer(bytes: inputB, length: byteCount, options: .storageModeShared),
          let out = device.makeBuffer(length: byteCount, options: .storageModeShared)
    else {
        throw TestSupportError.bufferAllocationFailed
    }

    return (
        Tensor(buffer: inA, shape: [inputA.count], dataType: .f32),
        Tensor(buffer: inB, shape: [inputB.count], dataType: .f32),
        Tensor(buffer: out, shape: [inputA.count], dataType: .f32)
    )
}

// MARK: - Output assertions

func assertOutput(in tensor: Tensor, expected: [Float]) {
    let result = tensor.buffer.contents().bindMemory(to: Float.self, capacity: expected.count)

    for index in expected.indices {
        #expect(result[index] == expected[index])
    }
}

// MARK: - CommitFeedbackBox

@available(macOS 26.0, iOS 26.0, *) final class CommitFeedbackBox: @unchecked Sendable {
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

        if let error {
            throw error
        }
    }

    // MARK: Private

    private let semaphore = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private var error: (any Error)?
}
