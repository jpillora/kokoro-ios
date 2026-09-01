// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "KokoroSwift",
  platforms: [
    .iOS(.v18), .macOS(.v15)
  ],
  products: [
    .library(
      name: "KokoroSwift",
      targets: ["KokoroSwift"]
    ),
  ],
  dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.29.1"),
    // .package(url: "https://github.com/mlalma/eSpeakNGSwift", from: "1.0.1"),
    // Fork of mlalma/MisakiSwift 1.0.6 with the library built static (no
    // type: .dynamic) so MLX links once when embedded in an iOS app.
    .package(url: "https://github.com/jpillora/MisakiSwift", branch: "kokoro-ios"),
    .package(url: "https://github.com/mlalma/MLXUtilsLibrary.git", exact: "0.0.6")
  ],
  targets: [
    .target(
      name: "KokoroSwift",
      dependencies: [
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
        .product(name: "MLXRandom", package: "mlx-swift"),
        .product(name: "MLXFFT", package: "mlx-swift"),
        // .product(name: "eSpeakNGLib", package: "eSpeakNGSwift"),
        .product(name: "MisakiSwift", package: "MisakiSwift"),
        .product(name: "MLXUtilsLibrary", package: "MLXUtilsLibrary")
      ],
      resources: [
       .copy("../../Resources/")
      ]
    ),
    .testTarget(
      name: "KokoroSwiftTests",
      dependencies: ["KokoroSwift"]
    ),
  ]
)
