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

    try encoder.encode(
        Kernels.Add(buffers.a, buffers.b, into: buffers.out),
        using: kernelLibrary
    )
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    try #require(commandBuffer.status == .completed)
    assertOutput(in: buffers.out, expected: expected)
}

@Test func addKernelMetal4() throws {
    guard #available(macOS 26.0, iOS 26.0, *) else { return }

    let device = try #require(MTLCreateSystemDefaultDevice())
    let commandQueue = try #require(device.makeMTL4CommandQueue())
    let allocator = try #require(device.makeCommandAllocator())
    let commandBuffer = try #require(device.makeCommandBuffer())
    commandBuffer.beginCommandBuffer(allocator: allocator)
    let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
    let buffers = try makeBuffers(device: device)

    let kernelLibrary = try KernelLibrary(device: device)
    let descriptor = MTL4ArgumentTableDescriptor()
    descriptor.maxBufferBindCount = 31
    descriptor.maxTextureBindCount = 0
    descriptor.maxSamplerStateBindCount = 0
    descriptor.initializeBindings = false
    descriptor.supportAttributeStrides = false
    let argumentTable = try device.makeArgumentTable(descriptor: descriptor)
    let constants = Metal4ConstantArena(device: device)

    try encoder.encode(
        Kernels.Add(buffers.a, buffers.b, into: buffers.out),
        using: kernelLibrary,
        argumentTable: argumentTable,
        constants: constants
    )

    encoder.endEncoding()
    commandBuffer.endCommandBuffer()

    let feedback = CommitFeedbackBox()
    let options = MTL4CommitOptions()
    options.addFeedbackHandler { commitFeedback in
        constants.reset()
        feedback.finish(error: commitFeedback.error)
    }

    commandQueue.commit([commandBuffer], options: options)
    try feedback.wait()

    assertOutput(in: buffers.out, expected: expected)
}
