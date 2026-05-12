import Dispatch
import Foundation

// MARK: - CommitFeedbackBox

final class CommitFeedbackBox: @unchecked Sendable {
    // MARK: Internal

    func finish(error: (any Error)?) {
        lock.lock()
        self.error = error
        lock.unlock()
        semaphore.signal()
    }

    func wait() throws {
        semaphore.wait()
        lock.lock()
        let error = error
        lock.unlock()
        if let error { throw error }
    }

    // MARK: Private

    private let semaphore = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private var error: (any Error)?
}
