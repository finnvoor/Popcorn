import Metal

// MARK: - Tensor

public struct Tensor {
    // MARK: Lifecycle

    public init(buffer: MTLBuffer, shape: Shape, dataType: DataType = .f32) {
        self.buffer = buffer
        self.shape = shape
        self.dataType = dataType
    }

    public init(buffer: MTLBuffer, shape: [Int], dataType: DataType = .f32) {
        self.init(buffer: buffer, shape: Shape(shape), dataType: dataType)
    }

    // MARK: Public

    public let buffer: MTLBuffer
    public let shape: Shape
    public let dataType: DataType

    // MARK: Internal

    @available(macOS 26.0, iOS 26.0, *) mutating func mtlTensorView() throws -> MTLTensor {
        if let mtlTensor = _mtlTensor as? MTLTensor {
            return mtlTensor
        } else {
            let descriptor = MTLTensorDescriptor()
            guard let dimensions = shape.mtlTensorExtents else {
                throw PopcornError.tensorShapeUnsupportedForMTLView
            }
            guard let dataType = dataType.mtlTensorDataType else {
                throw PopcornError.tensorDataTypeUnsupportedForMTLView(dataType)
            }
            descriptor.dimensions = dimensions
            descriptor.dataType = dataType
            descriptor.usage = .compute
            descriptor.resourceOptions = buffer.resourceOptions
            let mtlTensor = try buffer.makeTensor(descriptor: descriptor, offset: 0)
            _mtlTensor = mtlTensor
            return mtlTensor
        }
    }

    // MARK: Private

    private var _mtlTensor: MTLResource?
}

// MARK: - TensorElement

public protocol TensorElement: BitwiseCopyable {
    static var tensorDataType: Tensor.DataType { get }
}

// MARK: - UInt8 + TensorElement

extension UInt8: TensorElement { public static let tensorDataType = Tensor.DataType.u8 }

// MARK: - UInt16 + TensorElement

extension UInt16: TensorElement { public static let tensorDataType = Tensor.DataType.u16 }

// MARK: - UInt32 + TensorElement

extension UInt32: TensorElement { public static let tensorDataType = Tensor.DataType.u32 }

// MARK: - UInt64 + TensorElement

extension UInt64: TensorElement { public static let tensorDataType = Tensor.DataType.u64 }

// MARK: - Int8 + TensorElement

extension Int8: TensorElement { public static let tensorDataType = Tensor.DataType.i8 }

// MARK: - Int16 + TensorElement

extension Int16: TensorElement { public static let tensorDataType = Tensor.DataType.i16 }

// MARK: - Int32 + TensorElement

extension Int32: TensorElement { public static let tensorDataType = Tensor.DataType.i32 }

// MARK: - Int64 + TensorElement

extension Int64: TensorElement { public static let tensorDataType = Tensor.DataType.i64 }

// MARK: - Float + TensorElement

extension Float: TensorElement { public static let tensorDataType = Tensor.DataType.f32 }

// MARK: - Tensor.Shape

public extension Tensor {
    struct Shape: Sendable, Equatable, ExpressibleByArrayLiteral {
        // MARK: Lifecycle

        public init(_ dimensions: [Int]) {
            self.dimensions = dimensions
        }

        public init(arrayLiteral elements: Int...) {
            dimensions = elements
        }

        // MARK: Public

        public var dimensions: [Int]

        public var rank: Int {
            dimensions.count
        }

        public var elementCount: Int {
            dimensions.reduce(1, *)
        }

        public subscript(index: Int) -> Int {
            dimensions[index]
        }

        // MARK: Internal

        @available(macOS 26.0, iOS 26.0, *) var mtlTensorExtents: MTLTensorExtents? {
            MTLTensorExtents(dimensions)
        }
    }
}

// MARK: - Tensor.DataType

public extension Tensor {
    enum DataType: Sendable, Equatable {
        case u8
        case u16
        case u32
        case u64

        case i8
        case i16
        case i32
        case i64

        case f32
        case f16
        case bf16

        // MARK: Public

        public var stride: Int {
            switch self {
            case .u8, .i8: 1
            case .u16, .i16, .f16, .bf16: 2
            case .u32, .i32, .f32: 4
            case .u64, .i64: 8
            }
        }

        // MARK: Internal

        @available(macOS 26.0, iOS 26.0, *) var mtlTensorDataType: MTLTensorDataType? {
            switch self {
            case .u8: .uint8
            case .u16: .uint16
            case .u32: .uint32
            case .i8: .int8
            case .i16: .int16
            case .i32: .int32
            case .f16: .float16
            case .bf16: .bfloat16
            case .f32: .float32
            case .u64, .i64: nil
            }
        }
    }
}

// MARK: - Tensor.Access

public extension Tensor {
    struct Access: OptionSet, Sendable {
        // MARK: Lifecycle

        public init(rawValue: UInt8) {
            self.rawValue = rawValue
        }

        // MARK: Public

        public static let read = Tensor.Access(rawValue: 1 << 0)
        public static let write = Tensor.Access(rawValue: 1 << 1)

        public static let readWrite: Tensor.Access = [.read, .write]

        public let rawValue: UInt8
    }
}

// MARK: - Tensor.Binding

public extension Tensor {
    struct Binding {
        // MARK: Lifecycle

        public init(tensor: Tensor, access: Tensor.Access) {
            precondition(!access.isEmpty, "Tensor binding access must not be empty")
            self.tensor = tensor
            self.access = access
        }

        // MARK: Public

        public let tensor: Tensor
        public let access: Tensor.Access
    }
}
