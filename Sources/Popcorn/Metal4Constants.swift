import Metal

// MARK: - Metal4Constants

@available(macOS 26.0, iOS 26.0, *) public protocol Metal4Constants: AnyObject {
    func append(_ bytes: UnsafeRawBufferPointer, alignment: Int) throws -> MTLGPUAddress
}

@available(macOS 26.0, iOS 26.0, *) public extension Metal4Constants {
    func append<T: BitwiseCopyable>(_ value: T) throws -> MTLGPUAddress {
        try withUnsafeBytes(of: value) { bytes in
            try append(bytes, alignment: MemoryLayout<T>.alignment)
        }
    }
}

// MARK: - Metal4ConstantArena

@available(macOS 26.0, iOS 26.0, *) public final class Metal4ConstantArena: Metal4Constants, @unchecked Sendable {
    // MARK: Lifecycle

    public init(
        device: MTLDevice,
        residencySet: (any MTLResidencySet)? = nil,
        pageSize: Int = 64 * 1024
    ) {
        self.device = device
        self.residencySet = residencySet
        self.pageSize = pageSize
    }

    // MARK: Public

    public func append(_ bytes: UnsafeRawBufferPointer, alignment: Int) throws -> MTLGPUAddress {
        let byteCount = max(bytes.count, 1)
        let alignment = max(alignment, Self.minimumAlignment)

        while true {
            if pageIndex < pages.count {
                let page = pages[pageIndex]
                let alignedOffset = Self.align(offset, to: alignment)
                if alignedOffset + byteCount <= page.length {
                    if let source = bytes.baseAddress, !bytes.isEmpty {
                        page.contents()
                            .advanced(by: alignedOffset)
                            .copyMemory(from: source, byteCount: bytes.count)
                        // didModifyRange is only meaningful for managed
                        // storage. Our pages are shared+untracked, so skipping
                        // the call avoids the per-constant CFI/objc-msg cost.
                    }
                    offset = alignedOffset + byteCount
                    return page.gpuAddress + UInt64(alignedOffset)
                }

                pageIndex += 1
                offset = 0
                continue
            }

            try allocatePage(minLength: byteCount)
        }
    }

    public func reset() {
        pageIndex = 0
        offset = 0
    }

    public func preallocate(pageCount: Int) throws {
        while pages.count < pageCount {
            try allocatePage(minLength: 0)
        }
    }

    // MARK: Private

    private static let minimumAlignment = 16

    private let device: MTLDevice
    private let residencySet: (any MTLResidencySet)?
    private let pageSize: Int

    private var pages: [MTLBuffer] = []
    private var pageIndex = 0
    private var offset = 0

    private static func align(_ value: Int, to alignment: Int) -> Int {
        (value + alignment - 1) & ~(alignment - 1)
    }

    private func allocatePage(minLength: Int) throws {
        let length = max(pageSize, minLength)
        guard let buffer = device.makeBuffer(
            length: length,
            options: [.storageModeShared, .cpuCacheModeWriteCombined, .hazardTrackingModeUntracked]
        ) else {
            throw PopcornError.constantAllocationFailed(byteCount: length)
        }
        buffer.label = "Popcorn.constants.\(length)"
        pages.append(buffer)
        residencySet?.addAllocation(buffer)
    }
}
