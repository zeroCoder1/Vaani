// swift-tools-version: 5.9
//
//  Package.swift
//  Vaani
//
//  Created by Shrutesh Sharma on 20/09/2026.
//

import PackageDescription
import class Foundation.ProcessInfo

let package = Package(
    name: "Vaani",
    platforms: [.iOS(.v16), .macOS(.v14)],
    products: [
        .library(name: "Vaani", targets: ["Vaani"])
    ],
    dependencies: [
        .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager",
                 from: "1.24.2")
    ],
    targets: [
        .target(
            name: "Vaani",
            dependencies: [
                .product(name: "onnxruntime",
                         package: "onnxruntime-swift-package-manager")
            ]
        ),
        // Fixtures live in Fixtures/ at the repo root and are read from disk
        // rather than bundled, so the demo app and the tests share one copy.
        .testTarget(name: "VaaniTests", dependencies: ["Vaani"]),
    ]
)

// The DocC plugin is only needed to render documentation locally, so it is
// added on demand rather than making every consumer resolve it:
//
//     VAANI_DOCS=1 swift package generate-documentation --target Vaani
//
// Swift Package Index builds the docs from .spi.yml without it.
if ProcessInfo.processInfo.environment["VAANI_DOCS"] != nil {
    package.dependencies.append(
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.3.0"))
}
