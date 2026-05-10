import Foundation
import Metal

public final class KernelLibrary {
    // MARK: Lifecycle

    public convenience init(device: MTLDevice) throws {
        try self.init(device: device, bundle: .module)
    }

    init(device: MTLDevice, bundle: Bundle) throws {
        self.device = device
        library = try device.makeDefaultLibrary(bundle: bundle)
    }

    // MARK: Public

    public func pipelineState(for functionName: String) throws -> MTLComputePipelineState {
        lock.lock()
        defer { lock.unlock() }

        if let cached = cache[functionName] {
            return cached
        }

        let function: MTLFunction? = if let bundledFunction = library.makeFunction(name: functionName) {
            bundledFunction
        } else if functionName.hasPrefix("mpp_") {
            try mppLibrary().makeFunction(name: functionName)
        } else {
            nil
        }

        guard let function else { throw PopcornError.kernelFunctionNotFound(functionName) }

        let pipelineState = try device.makeComputePipelineState(function: function)
        cache[functionName] = pipelineState
        return pipelineState
    }

    // MARK: Internal

    let device: MTLDevice

    static func mppSource(bundle: Bundle) throws -> String {
        let urls = ["MPPMatmul", "MPPMatvec"].compactMap {
            bundle.url(forResource: $0, withExtension: "metal", subdirectory: "Metal4")
        }
        guard urls.count == 2 else {
            throw PopcornError.kernelFunctionNotFound("mpp_*")
        }
        return try urls.map { try String(contentsOf: $0, encoding: .utf8) }.joined(separator: "\n")
    }

    // MARK: Private

    private let library: MTLLibrary

    private var cache: [String: MTLComputePipelineState] = [:]
    private var _mppLibrary: MTLLibrary?
    private let lock = NSLock()

    private func mppLibrary() throws -> MTLLibrary {
        if let _mppLibrary { return _mppLibrary }
        guard #available(macOS 26.0, *) else {
            throw PopcornError.kernelFunctionNotFound("mpp_*")
        }
        let source = try Self.mppSource(bundle: .module)
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let compiled = try device.makeLibrary(source: source, options: options)
        _mppLibrary = compiled
        return compiled
    }
}
