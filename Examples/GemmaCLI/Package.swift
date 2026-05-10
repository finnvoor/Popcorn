// swift-tools-version: 6.3

import PackageDescription

let package = Package(
    name: "GemmaCLI",
    platforms: [.macOS(.v26)],
    products: [
        .executable(name: "gemma-cli", targets: ["GemmaCLI"]),
    ],
    dependencies: [
        .package(name: "Popcorn", path: "../.."),
        .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.3.2"),
        .package(url: "https://github.com/finnvoor/MTLSafeTensors.git", branch: "main"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.5.0"),
    ],
    targets: [
        .executableTarget(
            name: "GemmaCLI",
            dependencies: [
                "Popcorn",
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                "MTLSafeTensors",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
    ],
    swiftLanguageModes: [.v6]
)
