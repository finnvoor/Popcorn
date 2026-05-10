// swift-tools-version: 6.3

import PackageDescription

let package = Package(
    name: "Popcorn",
    platforms: [.macOS(.v14), .iOS(.v16)],
    products: [
        .library(name: "Popcorn", targets: ["Popcorn"])
    ],
    targets: [
        .target(
            name: "PopcornShaderTypes",
            publicHeadersPath: "."
        ),
        .target(
            name: "Popcorn",
            dependencies: ["PopcornShaderTypes"],
            resources: [
                .copy("Kernels/Metal4")
            ]
        ),
        .testTarget(name: "PopcornTests", dependencies: ["Popcorn", "PopcornShaderTypes"])
    ],
    swiftLanguageModes: [.v6]
)
