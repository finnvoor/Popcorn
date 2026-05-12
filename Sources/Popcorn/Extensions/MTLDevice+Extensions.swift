import Metal

public extension MTLDevice {
    /// Whether this device can run Popcorn's Metal Performance Primitives kernels.
    ///
    /// MPP kernels are written in MSL 4.0 and require runtime tensor support, both of
    /// which are gated on `MTLGPUFamilyMetal4`. On unsupported devices Popcorn falls
    /// back to its hand-written kernels.
    var supportsMPP: Bool {
        guard #available(macOS 26.0, iOS 26.0, *) else { return false }
        return supportsFamily(.metal4)
    }

    func makeTensor(
        shape: Tensor.Shape,
        dataType: Tensor.DataType = .f32,
        options: MTLResourceOptions = .storageModeShared,
        label: String? = nil
    ) throws -> Tensor {
        let byteCount = max(shape.elementCount * dataType.stride, 1)
        guard let buffer = makeBuffer(length: byteCount, options: options) else {
            throw PopcornError.tensorAllocationFailed(byteCount: byteCount)
        }
        if let label { buffer.label = label }
        return Tensor(buffer: buffer, shape: shape, dataType: dataType)
    }

    func makeTensor<Element: TensorElement>(
        values: [Element],
        shape: Tensor.Shape,
        options: MTLResourceOptions = .storageModeShared,
        label: String? = nil
    ) throws -> Tensor {
        precondition(values.count == shape.elementCount, "Tensor values count must match shape element count")
        let byteCount = max(values.count * MemoryLayout<Element>.stride, 1)
        let buffer = if values.isEmpty {
            makeBuffer(length: byteCount, options: options)
        } else {
            makeBuffer(bytes: values, length: byteCount, options: options)
        }
        guard let buffer else {
            throw PopcornError.tensorAllocationFailed(byteCount: byteCount)
        }
        if let label { buffer.label = label }
        return Tensor(buffer: buffer, shape: shape, dataType: Element.tensorDataType)
    }
}
