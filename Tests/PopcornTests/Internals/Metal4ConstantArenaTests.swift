import Metal
@testable import Popcorn
import Testing

// MARK: - FirstConstant

private struct FirstConstant { var value: UInt32 }

// MARK: - SecondConstant

private struct SecondConstant { var a: UInt32; var b: UInt32 }

@Test func metal4ConstantArenaAlignsConstants() throws {
    guard #available(macOS 26.0, iOS 26.0, *) else { return }

    let device = try #require(MTLCreateSystemDefaultDevice())
    let arena = Metal4ConstantArena(device: device, pageSize: 256)

    let first = try arena.append(FirstConstant(value: 1))
    let second = try arena.append(SecondConstant(a: 2, b: 3))

    #expect(second - first >= 16)
    #expect(second % 16 == 0)
}

@Test func metal4ConstantArenaResetReusesStorage() throws {
    guard #available(macOS 26.0, iOS 26.0, *) else { return }

    let device = try #require(MTLCreateSystemDefaultDevice())
    let arena = Metal4ConstantArena(device: device, pageSize: 256)

    let first = try arena.append(FirstConstant(value: 1))
    arena.reset()
    let reused = try arena.append(FirstConstant(value: 2))

    #expect(reused == first)
}
