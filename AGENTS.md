# AGENTS.md

## Build, Test, and Format

This project uses [mise](https://mise.jdx.dev/) for task running. Use the following commands:

- `mise build` — build the project
- `mise test` — run the test suite
- `mise format` — format the code (run this after making code changes)

Always use these `mise` tasks instead of invoking the underlying tools (e.g. `swift build`, `swift test`) directly. Run `mise format` after any code changes.

## API and Kernel Design Philosophy

Popcorn is a low-level tensor/kernel library, not a high-level inference engine. Keep the public API small, explicit, and Swifty while hiding backend/kernel-selection details from consumers.

- Prefer one semantic top-level kernel type per operation. For example, consumers should use `Kernels.Matmul` rather than choosing between matmul, matvec, MPP matmul, quantized matmul, or decode-specific variants.
- Kernel initializers should select the best implementation internally based on shapes, dtypes, hardware support, and storage format. Consumers should not need to pick dispatch sizes, threadgroup sizes, MPP vs non-MPP paths, or decode vs prefill kernels.
- Keep implementation-specific kernels private/internal when possible. Metal function names such as `aq_qmv_fast_*` or `mpp_matmul_*` are backend details, not public API concepts.
- Prefer overloads for alternate storage types over new operation names. For example, `Kernels.Matmul(_:_:into:)` can accept either dense `Tensor` weights or an `AffineQuantizedTensor` RHS.
- Keep Popcorn focused on reusable kernels and storage abstractions. Model-specific loader conveniences, dense-vs-quantized weight enums, tokenizer logic, and inference graph decisions belong in examples or downstream packages.
- Prefer throwing initializers/errors for invalid runtime metadata instead of `precondition` traps, especially for tensor shapes, quantization metadata, and model file contents.
- Optimize behind stable semantic APIs. It is fine for `Kernels.Matmul` to route to specialized matvec, MPP, quantized decode, or future tiled QMM implementations as long as the public operation remains coherent.
