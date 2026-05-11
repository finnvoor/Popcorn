import Metal

enum DispatchSize {
    // MARK: Internal

    static func reduction(
        rowCount: Int,
        n: Int,
        pipelineState: MTLComputePipelineState
    ) -> (grid: MTLSize, threadgroupSize: MTLSize) {
        let simdWidth = max(1, pipelineState.threadExecutionWidth)
        let maxThreadgroupWidth = max(
            simdWidth,
            roundDown(min(1024, pipelineState.maxTotalThreadsPerThreadgroup), toMultipleOf: simdWidth)
        )
        let preferredWidth = preferredThreadgroupWidth(
            rowCount: rowCount,
            n: n,
            simdWidth: simdWidth,
            maxThreadgroupWidth: maxThreadgroupWidth
        )
        let width = max(simdWidth, min(preferredWidth, roundUp(nextPowerOfTwo(n), toMultipleOf: simdWidth)))
        return (
            MTLSize(width: width * rowCount, height: 1, depth: 1),
            MTLSize(width: width, height: 1, depth: 1)
        )
    }

    static func rowsColumns(
        rowCount: Int,
        columnCount: Int,
        pipelineState: MTLComputePipelineState
    ) -> (grid: MTLSize, threadgroupSize: MTLSize) {
        columnsFirst(rowCount: rowCount, columnCount: columnCount, maxThreads: 256, pipelineState: pipelineState)
    }

    // MARK: Private

    private static func columnsFirst(
        rowCount: Int,
        columnCount: Int,
        maxThreads: Int,
        pipelineState: MTLComputePipelineState
    ) -> (grid: MTLSize, threadgroupSize: MTLSize) {
        let simdWidth = max(1, pipelineState.threadExecutionWidth)
        let maxThreads = max(
            simdWidth,
            roundDown(min(maxThreads, pipelineState.maxTotalThreadsPerThreadgroup), toMultipleOf: simdWidth)
        )

        let threadgroupHeight: Int
        let threadgroupWidth: Int
        if columnCount >= simdWidth {
            threadgroupHeight = max(simdWidth, roundUp(min(columnCount, maxThreads), toMultipleOf: simdWidth))
            threadgroupWidth = 1
        } else {
            threadgroupHeight = max(1, columnCount)
            threadgroupWidth = max(1, min(rowCount, maxThreads / threadgroupHeight))
        }

        return (
            MTLSize(width: rowCount, height: columnCount, depth: 1),
            MTLSize(width: threadgroupWidth, height: threadgroupHeight, depth: 1)
        )
    }

    private static func preferredThreadgroupWidth(
        rowCount: Int,
        n: Int,
        simdWidth: Int,
        maxThreadgroupWidth: Int
    ) -> Int {
        let simdgroupCount = if rowCount <= 4 {
            if n <= 256 { 8 }
            else if n <= 1_536 { 16 }
            else { 32 }
        } else if rowCount <= 16 {
            if n <= 256 { 4 }
            else if n <= 3_072 { 16 }
            else { 32 }
        } else if rowCount <= 64 {
            if n <= 768 { 4 }
            else if n <= 3_072 { 8 }
            else if n <= 8_192 { 16 }
            else { 32 }
        } else if rowCount <= 256 {
            if n <= 256 { 1 }
            else if n <= 3_072 { 2 }
            else if n <= 8_192 { 4 }
            else { 32 }
        } else if rowCount <= 512 {
            if n <= 768 { 1 }
            else if n <= 1_536 { 2 }
            else if n <= 3_072 { 16 }
            else if n <= 8_192 { 8 }
            else { 32 }
        } else {
            if n <= 768 { 2 }
            else if n <= 1_536 { 8 }
            else if n <= 3_072 { 32 }
            else { 16 }
        }

        return min(maxThreadgroupWidth, simdgroupCount * simdWidth)
    }

    private static func roundDown(_ value: Int, toMultipleOf multiple: Int) -> Int {
        value / multiple * multiple
    }

    private static func roundUp(_ value: Int, toMultipleOf multiple: Int) -> Int {
        ((value + multiple - 1) / multiple) * multiple
    }

    private static func nextPowerOfTwo(_ n: Int) -> Int {
        var v = 1
        while v < n {
            v <<= 1
        }
        return v
    }
}
