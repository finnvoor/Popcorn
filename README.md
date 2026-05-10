# Popcorn

Popcorn is a tiny Swift package for running optimized tensor kernels on Metal. It wraps Metal buffers as `Tensor`s, validates kernel shapes and data types, and provides encoder helpers for both classic Metal (`MTLComputeCommandEncoder`) and Metal 4 (`MTL4ComputeCommandEncoder`).

## Usage

Add Popcorn as a Swift Package dependency and import it with Metal:

```swift
import Metal
import Popcorn
```

Create tensors from Swift arrays or allocate empty tensors, make a `KernelLibrary`, then encode a kernel.

## Metal (`MTLComputeCommandEncoder`)

```swift
let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let commandBuffer = queue.makeCommandBuffer()!
let encoder = commandBuffer.makeComputeCommandEncoder()!
let library = try KernelLibrary(device: device)

let values: [Float] = [1, 2, 3, 4]
let a = try device.makeTensor(values: values, shape: [values.count], label: "a")
let b = try device.makeTensor(values: values, shape: [values.count], label: "b")
let out = try device.makeTensor(shape: [values.count], dataType: .f32, label: "out")

try encoder.encode(Kernels.Add(a, b, into: out), using: library)

encoder.endEncoding()
commandBuffer.commit()
commandBuffer.waitUntilCompleted()
```

## Metal 4 (`MTL4ComputeCommandEncoder`)

Popcorn's Metal 4 helper only binds kernel buffers/constants and dispatches threads. It does **not** manage residency, synchronization, command allocator reuse, hazard tracking, or constant-buffer lifetime beyond the `Metal4Constants` you provide. Add your own residency sets/barriers/lifetime management around this usage for real workloads.

```swift
let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeMTL4CommandQueue()!
let allocator = device.makeCommandAllocator()!
let commandBuffer = device.makeCommandBuffer()!
let library = try KernelLibrary(device: device)

let values: [Float] = [1, 2, 3, 4]
let a = try device.makeTensor(values: values, shape: [values.count], label: "a")
let b = try device.makeTensor(values: values, shape: [values.count], label: "b")
let out = try device.makeTensor(shape: [values.count], dataType: .f32, label: "out")

// For non-trivial workloads, create and maintain residency sets yourself, e.g.
// add all buffers/pipelines/argument tables/constants that must stay resident.
let residencyDescriptor = MTLResidencySetDescriptor()
residencyDescriptor.initialCapacity = 128
let residencySet = try device.makeResidencySet(descriptor: residencyDescriptor)
for buffer in [a.buffer, b.buffer, out.buffer] {
    residencySet.addAllocation(buffer)
}
residencySet.commit()
residencySet.requestResidency()
queue.addResidencySet(residencySet)

commandBuffer.beginCommandBuffer(allocator: allocator)
let encoder = commandBuffer.makeComputeCommandEncoder()!

let descriptor = MTL4ArgumentTableDescriptor()
descriptor.maxBufferBindCount = 31
descriptor.maxTextureBindCount = 0
descriptor.maxSamplerStateBindCount = 0
let argumentTable = try device.makeArgumentTable(descriptor: descriptor)
let constants = Metal4ConstantArena(device: device, residencySet: residencySet)

try encoder.encode(
    Kernels.Add(a, b, into: out),
    using: library,
    argumentTable: argumentTable,
    constants: constants
)

encoder.endEncoding()
commandBuffer.endCommandBuffer()
queue.commit([commandBuffer])
```
