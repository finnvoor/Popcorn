import Metal

// MARK: - ScratchSpec

/// A temporary tensor request made by composite kernels.
///
/// Popcorn only describes the temporary tensor's shape and element type. The
/// caller's `KernelCommandEncoder` decides how to allocate, pool, recycle, and
/// synchronize the underlying buffer.
public struct ScratchSpec {
    // MARK: Lifecycle

    public init(_ shape: [Int], _ dataType: Tensor.DataType) {
        self.shape = shape
        self.dataType = dataType
    }

    // MARK: Public

    public let shape: [Int]
    public let dataType: Tensor.DataType

    public var byteCount: Int {
        max(1, shape.reduce(1, *) * dataType.stride)
    }
}
