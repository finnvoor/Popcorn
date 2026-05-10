import Metal
import MTLSafeTensors
import Popcorn

// MARK: - GemmaError

enum GemmaError: Error, CustomStringConvertible {
    case message(String)

    // MARK: Internal

    var description: String {
        switch self { case let .message(message): message }
    }
}

extension UInt16 {
    var bf16Float: Float {
        Float(bitPattern: UInt32(self) << 16)
    }
}

extension MTLDevice {
    func makeNormWeightTensor(archive: SafeTensors, name: String) throws -> Tensor {
        let raw = try archive.tensor(named: name)
        guard raw.dtype == .bf16 else {
            return Tensor(buffer: raw.buffer, shape: raw.shape, dataType: raw.dtype.popcornDataType)
        }

        let count = raw.shape.reduce(1, *)
        let tensor = try makeTensor(shape: Tensor.Shape(raw.shape), dataType: .f32, label: name)
        let src = raw.buffer.contents().bindMemory(to: UInt16.self, capacity: count)
        let dst = tensor.buffer.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            dst[i] = src[i].bf16Float
        }
        return tensor
    }
}
