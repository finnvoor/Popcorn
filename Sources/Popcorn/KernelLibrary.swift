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

    public let device: MTLDevice

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

    static func mppSource(bundle: Bundle) throws -> String {
        let names = ["MPPMatmul", "MPPMatvec", "MPPFlashAttention"]
        let urls = names.compactMap {
            bundle.url(forResource: $0, withExtension: "metal", subdirectory: "Metal4")
        }
        guard urls.count == names.count else {
            throw PopcornError.kernelFunctionNotFound("mpp_*")
        }
        guard let headerURL = bundle.url(forResource: "PopcornKernelTypes", withExtension: "h") else {
            throw PopcornError.kernelFunctionNotFound("PopcornKernelTypes.h")
        }
        let header = try String(contentsOf: headerURL, encoding: .utf8)
        let mppFlashAttentionConstants = try Self.sharedType(
            named: "MPPFlashAttentionConstants",
            in: header
        )
        return try zip(names, urls).map { name, url in
            let source = try String(contentsOf: url, encoding: .utf8)
            return name == "MPPFlashAttention" ? mppFlashAttentionConstants + "\n" + source : source
        }.joined(separator: "\n")
    }

    static func sharedType(named name: String, in header: String) throws -> String {
        let marker = "} \(name);"
        guard let end = header.range(of: marker)?.upperBound else {
            throw PopcornError.kernelFunctionNotFound(name)
        }
        let prefix = header[..<end]
        guard let start = prefix.range(of: "typedef struct", options: .backwards)?.lowerBound else {
            throw PopcornError.kernelFunctionNotFound(name)
        }
        return String(header[start..<end])
    }

    // MARK: Private

    private let library: MTLLibrary

    private var cache: [String: MTLComputePipelineState] = [:]
    private var _mppLibrary: MTLLibrary?
    private let lock = NSLock()

    private func mppLibrary() throws -> MTLLibrary {
        if let _mppLibrary { return _mppLibrary }
        guard #available(macOS 26.0, iOS 26.0, *), device.supportsMPP else {
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
