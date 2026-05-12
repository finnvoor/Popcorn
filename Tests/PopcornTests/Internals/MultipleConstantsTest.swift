import Metal
@testable import Popcorn
import Testing

// MARK: - ShapeParams

private struct ShapeParams { var count: UInt32 }

// MARK: - FlagParams

private struct FlagParams { var scale: Float }

// MARK: - DummyKernel

private struct DummyKernel: DispatchKernel {
    let buffer: MTLBuffer
    let shape: ShapeParams
    let flags: FlagParams

    let functionName = "scalar_mul" // any real kernel; not actually dispatched.

    var tensors: [Tensor.Binding] {
        [.init(tensor: Tensor(buffer: buffer, shape: [1]), access: .read)]
    }

    var constants: [any BitwiseCopyable] {
        [shape, flags]
    }

    func dispatchSize(for _: MTLComputePipelineState) -> (grid: MTLSize, threadgroupSize: MTLSize) {
        (
            MTLSize(width: 1, height: 1, depth: 1),
            MTLSize(width: 1, height: 1, depth: 1)
        )
    }
}

@Test func kernelCanExposeMultipleConstantsBlocks() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    guard let buffer = device.makeBuffer(length: 16, options: .storageModeShared) else {
        throw TestSupportError.bufferAllocationFailed
    }
    let kernel = DummyKernel(
        buffer: buffer,
        shape: ShapeParams(count: 7),
        flags: FlagParams(scale: 1.5)
    )
    #expect(kernel.constants.count == 2)
}
