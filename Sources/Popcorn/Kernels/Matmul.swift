import Foundation
import Metal
import PopcornShaderTypes

public extension Kernels {
    struct Matmul: DispatchKernel {
        // MARK: Lifecycle

        public init(
            a: Tensor,
            b: Tensor,
            c: Tensor,
            m: Int,
            k: Int,
            n: Int,
            transposeB: Bool = false
        ) throws {
            self.a = a
            rhs = .dense(b)
            self.c = c

            if m == 1 {
                let configuration = try Self.matvecConfiguration(
                    x: a, w: b, out: c,
                    k: k, n: n, transposeW: transposeB
                )
                functionName = configuration.functionName
                constants = configuration.constants
                dispatchGrid = configuration.grid
                dispatchThreadgroupSize = configuration.threadgroupSize
                return
            }

            let device = a.buffer.device
            let useMPP = Self.supportsMPP(device, dataTypes: (a.dataType, b.dataType, c.dataType))
            let tileM = Self.pickMPPTileM(
                m: m,
                aDataType: a.dataType, bDataType: b.dataType, outDataType: c.dataType,
                transposeB: transposeB, useMPP: useMPP
            )

            functionName = try Self.functionName(
                aDataType: a.dataType, bDataType: b.dataType, outDataType: c.dataType,
                transposeB: transposeB, useMPP: useMPP, tileM: tileM
            )
            constants = [MatmulConstants(
                M: UInt32(m), K: UInt32(k), N: UInt32(n),
                transposeB: transposeB ? 1 : 0
            )]

            if useMPP {
                let tileN = 64
                let threadsPerTG = 128
                let mTiles = (m + tileM - 1) / tileM
                let nTiles = (n + tileN - 1) / tileN
                dispatchGrid = MTLSize(
                    width: nTiles * threadsPerTG,
                    height: mTiles,
                    depth: 1
                )
                dispatchThreadgroupSize = MTLSize(width: threadsPerTG, height: 1, depth: 1)
            } else {
                dispatchGrid = MTLSize(width: m, height: n, depth: 1)
                dispatchThreadgroupSize = MTLSize(width: 8, height: 8, depth: 1)
            }
        }

        public init(_ a: Tensor, _ b: Tensor, into c: Tensor, transposeB: Bool = false) throws {
            guard a.shape.rank == 2 else {
                throw PopcornError.tensorInvalidRank(expected: 2, actual: a.shape.rank)
            }
            guard b.shape.rank == 2 else {
                throw PopcornError.tensorInvalidRank(expected: 2, actual: b.shape.rank)
            }
            guard c.shape.rank == 2 else {
                throw PopcornError.tensorInvalidRank(expected: 2, actual: c.shape.rank)
            }

            let m = a.shape[0]
            let k = a.shape[1]
            let n = transposeB ? b.shape[0] : b.shape[1]
            let expectedB = transposeB ? [n, k] : [k, n]
            let expectedC = [m, n]

            guard b.shape.dimensions == expectedB else {
                throw PopcornError.tensorShapeMismatch("Matmul RHS shape mismatch: expected \(expectedB), got \(b.shape.dimensions).")
            }
            guard c.shape.dimensions == expectedC else {
                throw PopcornError.tensorShapeMismatch("Matmul output shape mismatch: expected \(expectedC), got \(c.shape.dimensions).")
            }

            try self.init(a: a, b: b, c: c, m: m, k: k, n: n, transposeB: transposeB)
        }

        /// Multiplies by a block-affine quantized RHS stored in linear-layer layout.
        ///
        /// `b` has logical shape `[outFeatures, inFeatures]`; this initializer computes
        /// `c = a @ b^T`, matching the dense `transposeB: true` linear-layer convention.
        public init(_ a: Tensor, _ b: AffineQuantizedTensor, into c: Tensor) throws {
            guard a.shape.rank == 2 else {
                throw PopcornError.tensorInvalidRank(expected: 2, actual: a.shape.rank)
            }
            guard c.shape.rank == 2 else {
                throw PopcornError.tensorInvalidRank(expected: 2, actual: c.shape.rank)
            }

            let m = a.shape[0]
            let k = a.shape[1]
            let n = b.outFeatures

            guard b.inFeatures == k else {
                throw PopcornError.tensorShapeMismatch(
                    "Matmul quantized RHS shape mismatch: a.shape=[\(m),\(k)] but b.inFeatures=\(b.inFeatures)."
                )
            }
            guard c.shape.dimensions == [m, n] else {
                throw PopcornError.tensorShapeMismatch(
                    "Matmul output shape mismatch: expected [\(m),\(n)], got \(c.shape.dimensions)."
                )
            }

            self.a = a
            rhs = .affineQuantized(packedValues: b.packedValues, scales: b.scales, biases: b.biases ?? b.scales)
            self.c = c

            let configuration = try Self.affineQuantizedConfiguration(x: a, w: b, out: c, m: m, k: k, n: n)
            functionName = configuration.functionName
            constants = configuration.constants
            dispatchGrid = configuration.grid
            dispatchThreadgroupSize = configuration.threadgroupSize
        }

        // MARK: Public

        public var functionName: String

        public var constants: [any BitwiseCopyable]

        public var tensors: [Tensor.Binding] {
            switch rhs {
            case let .dense(b):
                [
                    .init(tensor: a, access: .read),
                    .init(tensor: b, access: .read),
                    .init(tensor: c, access: .write)
                ]
            case let .affineQuantized(packedValues, scales, biases):
                [
                    .init(tensor: packedValues, access: .read),
                    .init(tensor: scales, access: .read),
                    .init(tensor: biases, access: .read),
                    .init(tensor: a, access: .read),
                    .init(tensor: c, access: .write)
                ]
            }
        }

        public func dispatchSize(for _: MTLComputePipelineState) -> (grid: MTLSize, threadgroupSize: MTLSize) {
            (dispatchGrid, dispatchThreadgroupSize)
        }

        // MARK: Private

        private enum RHS {
            case dense(Tensor)
            case affineQuantized(packedValues: Tensor, scales: Tensor, biases: Tensor)
        }

        private let a: Tensor
        private let rhs: RHS
        private let c: Tensor
        private let dispatchGrid: MTLSize
        private let dispatchThreadgroupSize: MTLSize

        private static func pickMPPTileM(
            m: Int,
            aDataType: Tensor.DataType,
            bDataType: Tensor.DataType,
            outDataType: Tensor.DataType,
            transposeB: Bool,
            useMPP: Bool
        ) -> Int {
            guard useMPP, transposeB else { return 64 }

            switch (aDataType, bDataType, outDataType) {
            case (.bf16, .bf16, .bf16), (.bf16, .bf16, .f32):
                break
            default:
                return 64
            }

            if m <= 8 { return 8 }
            if m <= 16 { return 16 }
            if m <= 32 { return 32 }
            return 64
        }

        private static func functionName(
            aDataType: Tensor.DataType,
            bDataType: Tensor.DataType,
            outDataType: Tensor.DataType,
            transposeB: Bool,
            useMPP: Bool,
            tileM: Int
        ) throws -> String {
            if useMPP {
                let base: String
                switch (aDataType, bDataType, outDataType) {
                case (.f32, .f32, .f32): base = "mpp_matmul_f32_f32_f32"
                case (.f16, .f16, .f16): base = "mpp_matmul_f16_f16_f16"
                case (.f16, .f16, .f32): base = "mpp_matmul_f16_f16_f32"
                case (.bf16, .bf16, .f32): base = "mpp_matmul_bf16_bf16_f32"
                case (.bf16, .bf16, .bf16): base = "mpp_matmul_bf16_bf16_bf16"
                case (.f32, .bf16, .f32): base = "mpp_matmul_f32_bf16_f32"
                case (.bf16, .f32, .f32): base = "mpp_matmul_bf16_f32_f32"
                default: throw PopcornError.unsupportedDataTypeCombination("Unsupported MPP matmul data type combination: \(aDataType), \(bDataType), \(outDataType).")
                }
                let tbSuffix = transposeB ? "_tb" : ""
                let tileSuffix = tileM == 64 ? "" : "_m\(tileM)"
                return base + tbSuffix + tileSuffix
            } else {
                switch (aDataType, bDataType, outDataType) {
                case (.f32, .f32, .f32): return "matmul"
                case (.f32, .f16, .f32): return "matmul_f16"
                case (.f32, .bf16, .f32): return "matmul_bf16"
                case (.bf16, .bf16, .f32): return "matmul_bf16_bf16_f32"
                case (.bf16, .bf16, .bf16): return "matmul_bf16_bf16_bf16"
                default: throw PopcornError.unsupportedDataTypeCombination("Unsupported matmul data type combination: \(aDataType), \(bDataType), \(outDataType).")
                }
            }
        }

        private static func matvecConfiguration(
            x: Tensor,
            w: Tensor,
            out: Tensor,
            k: Int,
            n: Int,
            transposeW: Bool
        ) throws -> (functionName: String, constants: [any BitwiseCopyable], grid: MTLSize, threadgroupSize: MTLSize) {
            let useOptimizedNK = transposeW
            if useOptimizedNK, let optimizedFunctionName = optimizedNKMatvecFunctionName(
                xDataType: x.dataType,
                wDataType: w.dataType,
                outDataType: out.dataType
            ) {
                let constants = [MatvecConstants(K: UInt32(k), N: UInt32(n), transposeW: 1)]
                let rowsPerSimdgroup = 4
                let grid = MTLSize(width: ((n + rowsPerSimdgroup - 1) / rowsPerSimdgroup) * 32, height: 1, depth: 1)
                let threadgroupSize = MTLSize(width: 128, height: 1, depth: 1)
                return (optimizedFunctionName, constants, grid, threadgroupSize)
            } else if supportsMPP(x.buffer.device, dataTypes: (x.dataType, w.dataType, out.dataType)) {
                let functionName = try mppMatvecFunctionName(
                    xDataType: x.dataType,
                    wDataType: w.dataType,
                    outDataType: out.dataType,
                    transposeW: transposeW
                )
                let constants = [MatvecConstants(K: UInt32(k), N: UInt32(n), transposeW: transposeW ? 1 : 0)]
                let tileN = 64
                let threadsPerTG = 32
                let nTiles = (n + tileN - 1) / tileN
                let grid = MTLSize(width: nTiles * threadsPerTG, height: 1, depth: 1)
                let threadgroupSize = MTLSize(width: threadsPerTG, height: 1, depth: 1)
                return (functionName, constants, grid, threadgroupSize)
            } else {
                let functionName = try fallbackMatvecFunctionName(
                    xDataType: x.dataType,
                    wDataType: w.dataType,
                    outDataType: out.dataType
                )
                let constants = [MatvecConstants(K: UInt32(k), N: UInt32(n), transposeW: transposeW ? 1 : 0)]
                let grid = MTLSize(width: n, height: 1, depth: 1)
                let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
                return (functionName, constants, grid, threadgroupSize)
            }
        }

        private static func affineQuantizedConfiguration(
            x: Tensor,
            w: AffineQuantizedTensor,
            out: Tensor,
            m: Int,
            k: Int,
            n: Int
        ) throws -> (functionName: String, constants: [any BitwiseCopyable], grid: MTLSize, threadgroupSize: MTLSize) {
            guard k % w.format.groupSize == 0 else {
                throw PopcornError.tensorShapeMismatch(
                    "Matmul quantized RHS: K=\(k) is not divisible by groupSize=\(w.format.groupSize)."
                )
            }
            guard supportsAffineQuantizedMatmul(format: w.format, xDataType: x.dataType, scaleDataType: w.scales.dataType, outDataType: out.dataType) else {
                throw PopcornError.unsupportedDataTypeCombination(
                    "Unsupported affine quantized matmul combination: format=\(w.format), x=\(x.dataType), scales=\(w.scales.dataType), out=\(out.dataType)."
                )
            }

            let xTag = dtypeTag(x.dataType)
            let sTag = dtypeTag(w.scales.dataType)
            let oTag = dtypeTag(out.dataType)
            let kernelPrefix = m == 1 ? "aq_qmv_fast" : "aq_matvec_simd4"
            let functionName = "\(kernelPrefix)_\(xTag)_\(sTag)_\(oTag)_b\(w.format.bits)_g\(w.format.groupSize)"

            let perWord = w.format.valuesPerPackedElement
            let constants = [AffineQMatmulConstants(
                M: UInt32(m),
                N: UInt32(n),
                K: UInt32(k),
                kGroups: UInt32(k / w.format.groupSize),
                wordsPerRow: UInt32(k / perWord),
                hasBias: w.biases == nil ? 0 : 1
            )]

            if m == 1 {
                let rowBlocks = (n + 7) / 8
                return (
                    functionName,
                    constants,
                    MTLSize(width: m * 64, height: rowBlocks, depth: 1),
                    MTLSize(width: 64, height: 1, depth: 1)
                )
            } else {
                let nBlocks = (n + 3) / 4
                let threadgroupWidth = min(128, max(32, nBlocks * 32))
                return (
                    functionName,
                    constants,
                    MTLSize(width: nBlocks * 32, height: m, depth: 1),
                    MTLSize(width: threadgroupWidth, height: 1, depth: 1)
                )
            }
        }

        private static func supportsAffineQuantizedMatmul(
            format: AffineQuantizationFormat,
            xDataType: Tensor.DataType,
            scaleDataType: Tensor.DataType,
            outDataType: Tensor.DataType
        ) -> Bool {
            guard format.bits == 4, format.groupSize == 64, format.packing == .uint32LittleEndian else {
                return false
            }
            switch (xDataType, scaleDataType, outDataType) {
            case (.bf16, .bf16, .bf16),
                 (.bf16, .bf16, .f32),
                 (.f32, .bf16, .f32):
                return true
            default:
                return false
            }
        }

        private static func dtypeTag(_ dt: Tensor.DataType) -> String {
            switch dt {
            case .f32: "f32"
            case .f16: "f16"
            case .bf16: "bf16"
            default: "unknown"
            }
        }

        private static func optimizedNKMatvecFunctionName(
            xDataType: Tensor.DataType,
            wDataType: Tensor.DataType,
            outDataType: Tensor.DataType
        ) -> String? {
            switch (xDataType, wDataType, outDataType) {
            case (.f32, .f32, .f32): "matvec_nk_simd_f32_f32_f32"
            case (.f32, .bf16, .f32): "matvec_nk_simd_f32_bf16_f32"
            case (.bf16, .bf16, .f32): "matvec_nk_simd_bf16_bf16_f32"
            case (.bf16, .bf16, .bf16): "matvec_nk_simd_bf16_bf16_bf16"
            default: nil
            }
        }

        private static func fallbackMatvecFunctionName(
            xDataType: Tensor.DataType,
            wDataType: Tensor.DataType,
            outDataType: Tensor.DataType
        ) throws -> String {
            switch (xDataType, wDataType, outDataType) {
            case (.f32, .f32, .f32): "matvec"
            case (.f32, .f16, .f32): "matvec_f16"
            case (.f32, .bf16, .f32): "matvec_bf16"
            case (.bf16, .bf16, .f32): "matvec_bf16_bf16_f32"
            case (.bf16, .bf16, .bf16): "matvec_bf16_bf16_bf16"
            default: throw PopcornError.unsupportedDataTypeCombination("Unsupported matvec data type combination: \(xDataType), \(wDataType), \(outDataType).")
            }
        }

        private static func mppMatvecFunctionName(
            xDataType: Tensor.DataType,
            wDataType: Tensor.DataType,
            outDataType: Tensor.DataType,
            transposeW: Bool
        ) throws -> String {
            let base: String = switch (xDataType, wDataType, outDataType) {
            case (.f32, .f32, .f32): "mpp_matvec_f32_f32_f32"
            case (.f16, .f16, .f16): "mpp_matvec_f16_f16_f16"
            case (.f16, .f16, .f32): "mpp_matvec_f16_f16_f32"
            case (.bf16, .bf16, .f32): "mpp_matvec_bf16_bf16_f32"
            case (.bf16, .bf16, .bf16): "mpp_matvec_bf16_bf16_bf16"
            case (.f32, .bf16, .f32): "mpp_matvec_f32_bf16_f32"
            case (.bf16, .f32, .f32): "mpp_matvec_bf16_f32_f32"
            default: throw PopcornError.unsupportedDataTypeCombination("Unsupported MPP matvec data type combination: \(xDataType), \(wDataType), \(outDataType).")
            }
            return transposeW ? base + "_tb" : base
        }

        private static func dataTypesSupportedByMPP(
            _ aDataType: Tensor.DataType,
            _ bDataType: Tensor.DataType,
            _ outDataType: Tensor.DataType
        ) -> Bool {
            switch (aDataType, bDataType, outDataType) {
            case (.f32, .f32, .f32),
                 (.f16, .f16, .f16),
                 (.f16, .f16, .f32),
                 (.bf16, .bf16, .f32),
                 (.bf16, .bf16, .bf16),
                 (.f32, .bf16, .f32),
                 (.bf16, .f32, .f32):
                true
            default:
                false
            }
        }

        private static func supportsMPP(
            _ device: any MTLDevice,
            dataTypes: (Tensor.DataType, Tensor.DataType, Tensor.DataType)
        ) -> Bool {
            device.supportsMPP && dataTypesSupportedByMPP(dataTypes.0, dataTypes.1, dataTypes.2)
        }
    }
}
