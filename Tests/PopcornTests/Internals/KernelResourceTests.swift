import Metal
@testable import Popcorn
import Testing

@Test func scratchSpecComputesByteCount() {
    #expect(ScratchSpec([2, 3, 4], .f32).byteCount == 96)
    #expect(ScratchSpec([], .bf16).byteCount == 2)
}

@Test func kernelLibraryExposesDeviceForUserEncoders() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let library = try KernelLibrary(device: device)
    #expect(library.device === device)
}
