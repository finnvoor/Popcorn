import ArgumentParser
import Metal
import Popcorn

// MARK: - KernelBackend

enum KernelBackend: String, CaseIterable, ExpressibleByArgument {
    case metal
    case metal4
}

// MARK: - KernelContext

protocol KernelContext: AnyObject {
    func addResidency(_ buffers: [MTLBuffer])
    func commitResidency()
    func preallocateScratch(_ specs: [ScratchSpec]) throws
    func submit(_ encode: (KernelCommandEncoder) throws -> Void) throws -> CommitFeedbackBox
}
