@preconcurrency import Metal

// MARK: - Metal4ArgumentTablePool

@available(macOS 26.0, iOS 26.0, *) final class Metal4ArgumentTablePool {
    // MARK: Lifecycle

    init(device: MTLDevice) {
        self.device = device
        descriptor.maxBufferBindCount = 31
        descriptor.maxTextureBindCount = 0
        descriptor.maxSamplerStateBindCount = 0
        descriptor.initializeBindings = false
        descriptor.supportAttributeStrides = false
    }

    // MARK: Internal

    func reset() {
        index = 0
    }

    func next() throws -> any MTL4ArgumentTable {
        if index == tables.count {
            try tables.append(device.makeArgumentTable(descriptor: descriptor))
        }
        defer { index += 1 }
        return tables[index]
    }

    // MARK: Private

    private let device: MTLDevice
    private let descriptor = MTL4ArgumentTableDescriptor()
    private var tables: [any MTL4ArgumentTable] = []
    private var index = 0
}
