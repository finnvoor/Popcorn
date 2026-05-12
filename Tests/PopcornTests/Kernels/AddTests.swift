import Metal
@testable import Popcorn
import Testing

private let expected = zip(inputA, inputB).map(+)

@Test func addKernelMetal() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let commandQueue = try #require(device.makeCommandQueue())
    let commandBuffer = try #require(commandQueue.makeCommandBuffer())
    let encoder = try #require(commandBuffer.makeComputeCommandEncoder())

    let buffers = try makeBuffers(device: device)
    let kernelLibrary = try KernelLibrary(device: device)

    let kernelEncoder = KernelCommandEncoder(compute: encoder, library: kernelLibrary, scratch: TestScratchAllocator(device: device))
    try kernelEncoder.encode(Kernels.Add(buffers.a, buffers.b, into: buffers.out))
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    try #require(commandBuffer.status == .completed)
    assertOutput(in: buffers.out, expected: expected)
}

@Test func addKernelWithResultBuilder() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let commandQueue = try #require(device.makeCommandQueue())
    let commandBuffer = try #require(commandQueue.makeCommandBuffer())
    let encoder = try #require(commandBuffer.makeComputeCommandEncoder())

    let buffers = try makeBuffers(device: device)
    let kernelLibrary = try KernelLibrary(device: device)

    let kernelEncoder = KernelCommandEncoder(compute: encoder, library: kernelLibrary, scratch: TestScratchAllocator(device: device))
    try kernelEncoder.encode {
        try Kernels.Add(buffers.a, buffers.b, into: buffers.out)
    }
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    try #require(commandBuffer.status == .completed)
    assertOutput(in: buffers.out, expected: expected)
}
