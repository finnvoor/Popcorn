import Foundation
@preconcurrency import Metal
import Popcorn

// MARK: - Metal4DispatchProfiler

/// Captures per-dispatch GPU timestamps via `MTL4CounterHeap` and writes them to a
/// JSONL file, one event per line. The output format is:
///
///     {"type":"meta","device":"M1","ts_unit":"ns","reference_ns":1234}
///     {"type":"event","name":"matmul_bf16_bf16_f32_tb_m8","cb":0,"start":1500,"end":3200}
///     ...
///
/// `start` and `end` are GPU timestamps in nanoseconds, relative to `reference_ns`
/// (the first observed timestamp, so values stay small and JSONL stays compact).
@available(macOS 26.0, iOS 26.0, *) final class Metal4DispatchProfiler {
    // MARK: Lifecycle

    init(device: MTLDevice, capacity: Int = 4096, outputURL: URL) throws {
        self.device = device
        self.capacity = capacity
        self.outputURL = outputURL

        let descriptor = MTL4CounterHeapDescriptor()
        descriptor.type = .timestamp
        descriptor.count = capacity
        heap = try device.makeCounterHeap(descriptor: descriptor)
        (heap as AnyObject).setValue("gemma4.profiler", forKey: "label")

        FileManager.default.createFile(atPath: outputURL.path, contents: nil)
        guard let handle = FileHandle(forWritingAtPath: outputURL.path) else {
            throw GemmaError.message("Could not open \(outputURL.path) for writing.")
        }
        outputHandle = handle

        let meta = "{\"type\":\"meta\",\"device\":\"\(device.name)\",\"ts_unit\":\"ns\"}\n"
        outputHandle.write(Data(meta.utf8))
    }

    deinit {
        try? outputHandle.close()
    }

    // MARK: Internal

    struct ResolvedSnapshot {
        fileprivate let cbIndex: Int
        fileprivate let startSlot: Int
        fileprivate let endSlot: Int
        fileprivate let dispatches: [InFlightDispatch]
    }

    var heapCapacity: Int { capacity }

    /// Called from `Metal4KernelContext.submit` before encoding a new CB. Resets the
    /// per-CB state so subsequent dispatch hook calls accumulate into a fresh
    /// in-flight record.
    func beginCommandBuffer(index: Int) {
        lock.lock()
        if head + Self.maxDispatchesPerCB * 2 > capacity {
            // Wrap to the start of the heap when we'd overflow. The previous CB has
            // resolved its slice already (we resolve on completion), so this is safe.
            head = 0
        }
        current = InFlightCB(cbIndex: index, startSlot: head, dispatches: [])
        lock.unlock()
    }

    /// Hook that `KernelCommandEncoder` invokes before/after every Metal 4 dispatch.
    func dispatchHook(
        kernelName: String,
        phase: Metal4DispatchPhase,
        encoder: any MTL4ComputeCommandEncoder
    ) {
        lock.lock()
        defer { lock.unlock() }
        guard current != nil else { return }
        let slot = head
        head += 1
        encoder.writeTimestamp(
            granularity: .precise,
            counterHeap: heap,
            index: slot
        )
        switch phase {
        case .willDispatch:
            current?.dispatches.append(InFlightDispatch(name: kernelName, beginSlot: slot))
        case .didDispatch:
            break // end slot = beginSlot + 1
        }
    }

    /// Called from `Metal4KernelContext.submit` after `endCommandBuffer` returns.
    /// Snapshots the in-flight record so it can be resolved on the GPU completion handler.
    func endCommandBuffer() -> ResolvedSnapshot? {
        lock.lock()
        defer { lock.unlock() }
        guard let cb = current else { return nil }
        current = nil
        return ResolvedSnapshot(cbIndex: cb.cbIndex, startSlot: cb.startSlot, endSlot: head, dispatches: cb.dispatches)
    }

    /// Called from the CB completion handler. Resolves the timestamp range and emits
    /// one JSONL event per dispatch.
    func flush(_ snapshot: ResolvedSnapshot) {
        guard snapshot.endSlot > snapshot.startSlot else { return }
        guard let data = try? heap.resolveCounterRange(snapshot.startSlot..<snapshot.endSlot) else { return }

        let timestamps = data.withUnsafeBytes { buffer -> [UInt64] in
            let count = buffer.count / MemoryLayout<MTL4TimestampHeapEntry>.stride
            return (0..<count).map { i in
                buffer.load(fromByteOffset: i * MemoryLayout<MTL4TimestampHeapEntry>.stride, as: MTL4TimestampHeapEntry.self).timestamp
            }
        }

        lock.lock()
        if referenceTimestamp == nil, let first = timestamps.first(where: { $0 != 0 }) {
            referenceTimestamp = first
            let line = "{\"type\":\"reference\",\"reference_ns\":\(first)}\n"
            outputHandle.write(Data(line.utf8))
        }
        let reference = referenceTimestamp ?? 0
        lock.unlock()

        var buffer = Data()
        buffer.reserveCapacity(snapshot.dispatches.count * 96)
        for dispatch in snapshot.dispatches {
            let beginIdx = dispatch.beginSlot - snapshot.startSlot
            let endIdx = beginIdx + 1
            guard endIdx < timestamps.count else { continue }
            let begin = timestamps[beginIdx]
            let end = timestamps[endIdx]
            // Skip entries the driver marked invalid (zero on M1 when the device drops samples).
            guard begin != 0, end != 0, end >= begin else { continue }
            let startNs = Int64(begin) - Int64(reference)
            let endNs = Int64(end) - Int64(reference)
            buffer.append(Data(
                "{\"type\":\"event\",\"name\":\"\(dispatch.name)\",\"cb\":\(snapshot.cbIndex),\"start\":\(startNs),\"end\":\(endNs)}\n".utf8
            ))
        }
        if !buffer.isEmpty {
            outputHandle.write(buffer)
        }
    }

    // MARK: Fileprivate

    fileprivate struct InFlightDispatch {
        let name: String
        let beginSlot: Int
    }

    // MARK: Private

    private struct InFlightCB {
        let cbIndex: Int
        let startSlot: Int
        var dispatches: [InFlightDispatch]
    }

    private static let maxDispatchesPerCB = 2048

    private let device: MTLDevice
    private let capacity: Int
    private let heap: any MTL4CounterHeap
    private let outputURL: URL
    private let outputHandle: FileHandle
    private let lock = NSLock()
    private var head: Int = 0
    private var current: InFlightCB?
    private var referenceTimestamp: UInt64?
}
