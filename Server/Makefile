# SwiftPM (command line) cannot compile MLX's Metal shaders — only
# Xcode/xcodebuild can (see mlx-swift README). MLX looks for `mlx.metallib`
# next to the executable at runtime, so we extract it from an xcodebuild and
# colocate it with swift-build binaries.

DD := .xcodebuild
PRODUCTS := $(DD)/Build/Products/Release
METALLIB := $(PRODUCTS)/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib

.PHONY: metallib dev server demo dist clean

# One-time (and after every mlx-swift version bump): compile Metal shaders
# via xcodebuild and colocate the metallib for `swift build` / `swift run`.
metallib:
	xcodebuild -scheme KokoroServer -configuration Release \
	  -destination platform=macOS -derivedDataPath $(DD) -quiet build
	mkdir -p .build/debug .build/release
	cp $(METALLIB) .build/debug/mlx.metallib
	cp $(METALLIB) .build/release/mlx.metallib

dev:
	swift build
	@test -f .build/debug/mlx.metallib || $(MAKE) metallib

server: dev
	.build/debug/KokoroServer

demo: dev
	.build/debug/KokoroDemo

# Everything the binary needs beside it at runtime (SwiftPM resource
# bundles: model config + G2P dictionaries, plus the GPU kernels) gets
# tarred up and embedded as a Mach-O section; the server extracts it to
# ~/.local/state/kokoro-apple and re-execs from there.
EMBED := -Xlinker -sectcreate -Xlinker __DATA -Xlinker __kokoro_res \
         -Xlinker .build/release/kokoro-resources.tgz

# Self-contained distributable: single binary, resources embedded.
# Two passes: the first build produces the resource bundles, the second
# relinks with the archive of them embedded.
dist:
	$(MAKE) metallib
	swift build -c release --product KokoroServer
	# `metallib`'s copy lands in a plain .build/release directory, which the
	# release build above then replaces with a symlink to
	# .build/arm64-apple-macosx/release — taking the metallib with it. Re-copy
	# once the symlink exists, or `dist` fails on any clean .build.
	cp $(METALLIB) .build/release/mlx.metallib
	rm -f .build/release/kokoro-resources.tgz
	cd .build/release && tar -czf kokoro-resources.tgz *.bundle mlx.metallib
	swift build -c release --product KokoroServer $(EMBED)
	mkdir -p dist
	cp .build/release/KokoroServer dist/kokoro-server
	# Replace the linker's minimal ad-hoc signature: the kernel SIGKILLs
	# linker-signed binaries after the embed relink, despite codesign
	# --verify passing.
	codesign -f -s - dist/kokoro-server
	@echo "dist/kokoro-server ready (self-contained)"

clean:
	rm -rf .build $(DD) dist
