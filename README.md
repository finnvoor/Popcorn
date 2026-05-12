# Popcorn
Popcorn is a small Swift package of optimized Metal tensor kernels. It wraps Metal buffers as `Tensor`s, validates kernel shapes and data types, picks the concrete Metal function, computes dispatch geometry, and handles dependency hazard tracking.

Popcorn does **not** own command buffers, Metal 4 argument-table pools, constant arenas, scratch pools, residency sets, profiling, or scheduling. Those policies belong in the runtime/inference layer above Popcorn.

## Usage
```swift
import Metal
import Popcorn

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let commandBuffer = queue.makeCommandBuffer()!
let compute = commandBuffer.makeComputeCommandEncoder()!
let library = try KernelLibrary(device: device)

// Minimal scratch allocator that just allocates a new private buffer per request.
// Not the most efficient choice for inference workloads (no pooling/reuse, every
// composite kernel pays for a fresh allocation), but it is the simplest correct
// implementation. Real runtimes should back this with a pool.
final class ScratchPool: KernelScratchAllocator {
    init(device: MTLDevice) {
        self.device = device
    }

    func borrowTemporaryBuffer(length: Int) throws -> KernelTemporaryBuffer {
        let buffer = device.makeBuffer(length: max(1, length), options: [.storageModePrivate])!
        return KernelTemporaryBuffer(buffer: buffer)
    }

    private let device: MTLDevice
}
let scratchPool = ScratchPool(device: device)

let values: [Float] = [1, 2, 3, 4]
let a = try device.makeTensor(values: values, shape: [values.count], label: "a")
let b = try device.makeTensor(values: values, shape: [values.count], label: "b")
let out = try device.makeTensor(shape: [values.count], dataType: .f32, label: "out")

let encoder = KernelCommandEncoder(compute: compute, library: library, scratch: scratchPool)
try encoder.encode(Kernels.Add(a, b, into: out))

compute.endEncoding()
commandBuffer.commit()
commandBuffer.waitUntilCompleted()
```

## Sequencing kernels
`KernelCommandEncoder` has a result-builder convenience for explicit kernel sequences:

```swift
try encoder.encode {
    try Kernels.Matmul(x, w, into: y, transposeB: true)
    try Kernels.RMSNorm(y, weight: norm, into: z)
    try Kernels.FlashAttention(q: q, k: k, v: v, into: out, scale: scale)
}
```

## Custom scratch pools
Every `KernelCommandEncoder` requires a scratch allocator so temporary-buffer ownership is always explicit. A real runtime should back this with a pool that reuses buffers across encoders, optionally driven by a residency set; set `wasReused: true` when handing back a buffer that may still contain earlier writes so Popcorn's hazard tracker inserts a barrier before reading it.

## Metal 4
For Metal 4, Popcorn still performs the dispatch and hazard tracking. Your runtime supplies reusable resources — an argument table, a constants buffer, and scratch — via `Metal4KernelResourceProvider`, and manages the residency set itself.

Minimal end-to-end Metal 4 example, one `Kernels.Add` dispatch:

```swift
import Metal
import Popcorn

@available(macOS 26.0, *)
final class Resources: Metal4KernelResourceProvider {
    init(device: MTLDevice, residencySet: any MTLResidencySet) throws {
        self.device = device
        self.residencySet = residencySet

        let descriptor = MTL4ArgumentTableDescriptor()
        descriptor.maxBufferBindCount = 31
        argumentTable = try device.makeArgumentTable(descriptor: descriptor)

        constants = device.makeBuffer(length: 64 * 1024, options: [.storageModeShared, .cpuCacheModeWriteCombined])!
        residencySet.addAllocation(constants)
    }

    func nextArgumentTable() throws -> any MTL4ArgumentTable {
        argumentTable
    }

    func appendConstant(_ bytes: UnsafeRawBufferPointer, alignment: Int) throws -> MTLGPUAddress {
        let aligned = (constantsOffset + alignment - 1) & ~(alignment - 1)
        precondition(aligned + bytes.count <= constants.length, "constants page exhausted")
        constants.contents().advanced(by: aligned).copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
        constantsOffset = aligned + bytes.count
        return constants.gpuAddress + UInt64(aligned)
    }

    func borrowTemporaryBuffer(length: Int) throws -> KernelTemporaryBuffer {
        // Same caveat as the classic scratch example — swap in a real pool for real workloads.
        let buffer = device.makeBuffer(length: max(1, length), options: [.storageModePrivate])!
        residencySet.addAllocation(buffer)
        return KernelTemporaryBuffer(buffer: buffer)
    }

    func reset() {
        constantsOffset = 0
    }

    private let device: MTLDevice
    private let residencySet: any MTLResidencySet
    private let argumentTable: any MTL4ArgumentTable
    private let constants: MTLBuffer
    private var constantsOffset = 0
}

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeMTL4CommandQueue()!
let allocator = device.makeCommandAllocator()!
let library = try KernelLibrary(device: device)

let residencyDescriptor = MTLResidencySetDescriptor()
residencyDescriptor.initialCapacity = 16
let residencySet = try device.makeResidencySet(descriptor: residencyDescriptor)

let values: [Float] = [1, 2, 3, 4]
let a = try device.makeTensor(values: values, shape: [values.count], label: "a")
let b = try device.makeTensor(values: values, shape: [values.count], label: "b")
let out = try device.makeTensor(shape: [values.count], dataType: .f32, label: "out")
for buffer in [a.buffer, b.buffer, out.buffer] {
    residencySet.addAllocation(buffer)
}

let resources = try Resources(device: device, residencySet: residencySet)
residencySet.commit()
residencySet.requestResidency()
queue.addResidencySet(residencySet)

let commandBuffer = device.makeCommandBuffer()!
commandBuffer.beginCommandBuffer(allocator: allocator)
let compute = commandBuffer.makeComputeCommandEncoder()!

let encoder = try KernelCommandEncoder(compute: compute, library: library, resources: resources)
try encoder.encode(Kernels.Add(a, b, into: out))

compute.endEncoding()
commandBuffer.endCommandBuffer()

let options = MTL4CommitOptions()
options.addFeedbackHandler { _ in
    resources.reset()
}
queue.commit([commandBuffer], options: options)
```

Residency, argument-table reuse, constants-arena lifetime, and command buffer/allocator reuse are all caller policy. Real runtimes typically pool argument tables and grow the constants buffer in pages; the example above keeps it flat for clarity.