import Metal
@testable import Popcorn
import Testing

@Test func kernelLibraryCachesPipelineStates() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let kernelLibrary = try KernelLibrary(device: device)

    let first = try kernelLibrary.pipelineState(for: "add")
    let second = try kernelLibrary.pipelineState(for: "add")

    #expect(first === second)
}

@Test func kernelLibraryThrowsForUnknownFunction() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let kernelLibrary = try KernelLibrary(device: device)

    #expect(throws: PopcornError.kernelFunctionNotFound("does_not_exist")) {
        _ = try kernelLibrary.pipelineState(for: "does_not_exist")
    }
}
