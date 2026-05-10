import Foundation
import Metal

// MARK: - PopcornError

public struct PopcornError: Error, LocalizedError, CustomDebugStringConvertible, CustomNSError, Equatable {
    // MARK: Lifecycle

    init(
        code: Code,
        localizedDescription: String,
        debugDescription: String? = nil,
        userInfo: [String: any Sendable] = [:]
    ) {
        self.code = code
        _localizedDescription = localizedDescription
        _debugDescription = debugDescription ?? "PopcornError.\(code)"
        _userInfo = userInfo
    }

    // MARK: Public

    public enum Code: Int, Sendable {
        case constantAllocationFailed = -1001
        case tensorAllocationFailed = -1003
        case tensorShapeUnsupportedForMTLView = -1004
        case tensorDataTypeUnsupportedForMTLView = -1005
        case kernelFunctionNotFound = -1006
        case tensorShapeMismatch = -1007
        case tensorInvalidRank = -1008
        case tensorInvalidAxis = -1009
        case unsupportedDataTypeCombination = -1010
        case heapAllocationFailed = -1011
        case workspaceExhausted = -1012
        case gpuOutOfMemory = -1013
        case gpuTimeout = -1014
        case gpuPageFault = -1015
        case gpuInvalidResource = -1016
        case gpuDeviceRemoved = -1017
        case gpuAccessRevoked = -1018
        case gpuInternal = -1019
    }

    public static let errorDomain = "PopcornError"

    public let code: Code

    public var errorDescription: String? {
        _localizedDescription
    }

    public var debugDescription: String {
        _debugDescription
    }

    public var description: String {
        _localizedDescription
    }

    public var errorCode: Int {
        code.rawValue
    }

    public var errorUserInfo: [String: Any] {
        _userInfo
    }

    public static func == (lhs: PopcornError, rhs: PopcornError) -> Bool {
        lhs.code == rhs.code
    }

    // MARK: Private

    private let _localizedDescription: String
    private let _debugDescription: String
    private let _userInfo: [String: any Sendable]
}

public extension PopcornError {
    static func constantAllocationFailed(byteCount: Int) -> PopcornError {
        PopcornError(
            code: .constantAllocationFailed,
            localizedDescription: "Failed to allocate a constants buffer.",
            debugDescription: "PopcornError.constantAllocationFailed(byteCount: \(byteCount))",
            userInfo: ["byteCount": byteCount]
        )
    }

    static func tensorAllocationFailed(byteCount: Int) -> PopcornError {
        PopcornError(
            code: .tensorAllocationFailed,
            localizedDescription: "Failed to allocate a tensor buffer.",
            debugDescription: "PopcornError.tensorAllocationFailed(byteCount: \(byteCount))",
            userInfo: ["byteCount": byteCount]
        )
    }

    static var tensorShapeUnsupportedForMTLView: PopcornError {
        PopcornError(
            code: .tensorShapeUnsupportedForMTLView,
            localizedDescription: "Tensor shape is unsupported for a Metal tensor view."
        )
    }

    static func tensorDataTypeUnsupportedForMTLView(_ dataType: Tensor.DataType) -> PopcornError {
        PopcornError(
            code: .tensorDataTypeUnsupportedForMTLView,
            localizedDescription: "Tensor data type is unsupported for a Metal tensor view.",
            debugDescription: "PopcornError.tensorDataTypeUnsupportedForMTLView(\(dataType))",
            userInfo: ["dataType": String(describing: dataType)]
        )
    }

    static func kernelFunctionNotFound(_ functionName: String) -> PopcornError {
        PopcornError(
            code: .kernelFunctionNotFound,
            localizedDescription: "Kernel function was not found in the Metal library.",
            debugDescription: "PopcornError.kernelFunctionNotFound(\(functionName))",
            userInfo: ["functionName": functionName]
        )
    }

    static func tensorShapeMismatch(_ message: String) -> PopcornError {
        PopcornError(
            code: .tensorShapeMismatch,
            localizedDescription: message,
            debugDescription: "PopcornError.tensorShapeMismatch(\(message))"
        )
    }

    static func tensorInvalidRank(expected: Int, actual: Int) -> PopcornError {
        PopcornError(
            code: .tensorInvalidRank,
            localizedDescription: "Tensor rank mismatch: expected \(expected), got \(actual).",
            debugDescription: "PopcornError.tensorInvalidRank(expected: \(expected), actual: \(actual))",
            userInfo: ["expected": expected, "actual": actual]
        )
    }

    static func tensorInvalidAxis(_ axis: Int, rank: Int) -> PopcornError {
        PopcornError(
            code: .tensorInvalidAxis,
            localizedDescription: "Tensor axis \(axis) is invalid for rank \(rank).",
            debugDescription: "PopcornError.tensorInvalidAxis(\(axis), rank: \(rank))",
            userInfo: ["axis": axis, "rank": rank]
        )
    }

    static func unsupportedDataTypeCombination(_ message: String) -> PopcornError {
        PopcornError(
            code: .unsupportedDataTypeCombination,
            localizedDescription: message,
            debugDescription: "PopcornError.unsupportedDataTypeCombination(\(message))"
        )
    }

    static func heapAllocationFailed(requested: Int, recommendedMaxWorkingSet: UInt64) -> PopcornError {
        PopcornError(
            code: .heapAllocationFailed,
            localizedDescription: "Failed to allocate a Metal heap of \(requested) bytes (device's recommended max working set is \(recommendedMaxWorkingSet) bytes).",
            debugDescription: "PopcornError.heapAllocationFailed(requested: \(requested), recommendedMaxWorkingSet: \(recommendedMaxWorkingSet))",
            userInfo: ["requested": requested, "recommendedMaxWorkingSet": recommendedMaxWorkingSet]
        )
    }

    static func workspaceExhausted(requested: Int, used: Int, capacity: Int) -> PopcornError {
        PopcornError(
            code: .workspaceExhausted,
            localizedDescription: "Workspace heap exhausted: requested \(requested) bytes after \(used) used of \(capacity).",
            debugDescription: "PopcornError.workspaceExhausted(requested: \(requested), used: \(used), capacity: \(capacity))",
            userInfo: ["requested": requested, "used": used, "capacity": capacity]
        )
    }

    static func gpuOutOfMemory(detail: String) -> PopcornError {
        PopcornError(
            code: .gpuOutOfMemory,
            localizedDescription: "GPU ran out of memory while executing a command buffer: \(detail)",
            debugDescription: "PopcornError.gpuOutOfMemory(\(detail))",
            userInfo: ["detail": detail]
        )
    }

    static func gpuTimeout(detail: String) -> PopcornError {
        PopcornError(
            code: .gpuTimeout,
            localizedDescription: "GPU command buffer timed out: \(detail)",
            debugDescription: "PopcornError.gpuTimeout(\(detail))",
            userInfo: ["detail": detail]
        )
    }

    static func gpuPageFault(detail: String) -> PopcornError {
        PopcornError(
            code: .gpuPageFault,
            localizedDescription: "GPU page fault while executing a command buffer: \(detail)",
            debugDescription: "PopcornError.gpuPageFault(\(detail))",
            userInfo: ["detail": detail]
        )
    }

    static func gpuInvalidResource(detail: String) -> PopcornError {
        PopcornError(
            code: .gpuInvalidResource,
            localizedDescription: "GPU command buffer referenced an invalid resource: \(detail)",
            debugDescription: "PopcornError.gpuInvalidResource(\(detail))",
            userInfo: ["detail": detail]
        )
    }

    static func gpuDeviceRemoved(detail: String) -> PopcornError {
        PopcornError(
            code: .gpuDeviceRemoved,
            localizedDescription: "GPU device was removed: \(detail)",
            debugDescription: "PopcornError.gpuDeviceRemoved(\(detail))",
            userInfo: ["detail": detail]
        )
    }

    static func gpuAccessRevoked(detail: String) -> PopcornError {
        PopcornError(
            code: .gpuAccessRevoked,
            localizedDescription: "GPU access was revoked: \(detail)",
            debugDescription: "PopcornError.gpuAccessRevoked(\(detail))",
            userInfo: ["detail": detail]
        )
    }

    static func gpuInternal(detail: String) -> PopcornError {
        PopcornError(
            code: .gpuInternal,
            localizedDescription: "GPU command buffer reported an internal error: \(detail)",
            debugDescription: "PopcornError.gpuInternal(\(detail))",
            userInfo: ["detail": detail]
        )
    }
}

public extension PopcornError {
    static func fromMetalError(_ error: any Error) -> PopcornError? {
        let nsError = error as NSError
        let detail = nsError.localizedDescription
        if nsError.domain == MTLCommandBufferErrorDomain {
            switch MTLCommandBufferError.Code(rawValue: UInt(nsError.code)) {
            case .outOfMemory: return .gpuOutOfMemory(detail: detail)
            case .timeout: return .gpuTimeout(detail: detail)
            case .pageFault: return .gpuPageFault(detail: detail)
            case .invalidResource: return .gpuInvalidResource(detail: detail)
            case .deviceRemoved: return .gpuDeviceRemoved(detail: detail)
            case .accessRevoked: return .gpuAccessRevoked(detail: detail)
            default: return .gpuInternal(detail: detail)
            }
        }

        if #available(macOS 26.0, iOS 26.0, *), nsError.domain == MTL4CommandQueueErrorDomain {
            switch MTL4CommandQueueError.Code(rawValue: nsError.code) {
            case .outOfMemory: return .gpuOutOfMemory(detail: detail)
            case .timeout: return .gpuTimeout(detail: detail)
            case .deviceRemoved: return .gpuDeviceRemoved(detail: detail)
            case .accessRevoked: return .gpuAccessRevoked(detail: detail)
            default: return .gpuInternal(detail: detail)
            }
        }
        return nil
    }
}

public extension PopcornError.Code {
    static func ~= (code: PopcornError.Code, error: any Error) -> Bool {
        (error as? PopcornError)?.code == code
    }
}
