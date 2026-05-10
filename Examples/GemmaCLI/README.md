# GemmaCLI

slop gemma inference using [`swift-huggingface`](https://github.com/huggingface/swift-huggingface) to download model, [`swift-transformers`](https://github.com/huggingface/swift-transformers) to tokenize, [`MTLSafeTensors`](https://github.com/finnvoor/MTLSafeTensors) to memory-map safetensors to `MTLBuffer`s, and Popcorn kernels. 

## Usage

```bash
swift run -c release --build-system swiftbuild gemma-cli --prompt "The capital of Ireland is" --max-new 8
```
