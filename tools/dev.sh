#!/usr/bin/env bash
# Build/test helpers.
#
# NOTE: this repo lives under ~/Documents, which is iCloud-synced. iCloud stamps
# com.apple.FinderInfo onto files, and codesign refuses to sign a bundle that
# carries it ("resource fork, Finder information, or similar detritus not
# allowed"). SwiftPM's in-tree .build therefore fails to sign test bundles, so
# every command here builds to a scratch directory outside the synced tree.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="${INDICASR_SCRATCH:-${TMPDIR:-/tmp}/indicasr-build}"
DD="$SCRATCH/dd"
mkdir -p "$SCRATCH" "$DD"

case "${1:-help}" in
  test)
    cd "$REPO"
    swift test -c "${2:-debug}" --scratch-path "$SCRATCH/spm"
    ;;
  build)
    cd "$REPO"
    swift build -c "${2:-debug}" --scratch-path "$SCRATCH/spm"
    ;;
  app)
    cd "$REPO/Examples/IndicASRDemo"
    DEVELOPMENT_TEAM="${DEVELOPMENT_TEAM:-}" xcodegen generate
    xcodebuild -project IndicASRDemo.xcodeproj -scheme IndicASRDemo \
      -destination "platform=iOS Simulator,name=${SIM:-iPhone 17}" \
      -derivedDataPath "$DD" CODE_SIGNING_ALLOWED=NO build
    ;;
  run)
    "$0" app
    APP="$(find "$DD/Build/Products" -name IndicASRDemo.app -maxdepth 3 | head -1)"
    xcrun simctl boot "${SIM:-iPhone 17}" 2>/dev/null || true
    xcrun simctl install "${SIM:-iPhone 17}" "$APP"
    xcrun simctl launch "${SIM:-iPhone 17}" com.indicasr.demo
    ;;
  xcode)
    # Reset Xcode state and open the ONE thing that is safe to open.
    #
    # The demo project references the repo root as a local Swift package. If
    # Xcode also has the repo root open as a Folder, it refuses to treat the
    # same directory as both and the package product fails to resolve
    # ("Missing package product 'IndicASR'"). So: open the .xcodeproj, never
    # the folder.
    if pgrep -x Xcode >/dev/null; then
      echo "Xcode is running. Quit it first (Cmd-Q, not just closing the window),"
      echo "then run this again - it holds the stale project graph in memory."
      exit 1
    fi
    rm -rf "$REPO/.swiftpm"
    rm -rf "$REPO/Examples/IndicASRDemo/IndicASRDemo.xcodeproj/project.xcworkspace/xcshareddata/swiftpm"
    rm -rf ~/Library/Developer/Xcode/DerivedData/IndicASRDemo-*
    cd "$REPO/Examples/IndicASRDemo" && DEVELOPMENT_TEAM="${DEVELOPMENT_TEAM:-}" xcodegen generate >/dev/null
    echo "reset. opening IndicASRDemo.xcodeproj"
    open "$REPO/Examples/IndicASRDemo/IndicASRDemo.xcodeproj"
    ;;
  push-model)
    # Side-load the model straight into the simulator app container, so the app
    # runs with no server and no download at all. Uses APFS clones where it can,
    # so ~900MB copies in about a second and costs no extra disk.
    SIMNAME="${SIM:-iPhone 17}"
    SRC="$REPO/models/${2:-int8_nc}"
    [ -f "$SRC/manifest.json" ] || { echo "no manifest.json in $SRC — run tools/make_manifest.py"; exit 1; }
    xcrun simctl boot "$SIMNAME" 2>/dev/null || true
    CONTAINER="$(xcrun simctl get_app_container "$SIMNAME" com.indicasr.demo data)"       || { echo "app not installed — run tools/dev.sh run first"; exit 1; }
    DEST="$CONTAINER/Library/Application Support/IndicASR"
    mkdir -p "$DEST"
    for f in "$SRC"/*; do
      cp -c "$f" "$DEST/" 2>/dev/null || cp "$f" "$DEST/"
    done
    echo "pushed $(ls "$DEST" | wc -l | tr -d ' ') files -> $DEST"
    du -sh "$DEST"
    ;;
  stage)
    # Assemble exactly what a host needs, with checksums, into models/upload/.
    "$REPO/.venv/bin/python" "$REPO/tools/make_manifest.py" --src "$REPO/models/${2:-int8_nc}"
    "$REPO/.venv/bin/python" "$REPO/tools/stage_upload.py" --src "$REPO/models/${2:-int8_nc}"
    ;;
  serve)
    # Bind all interfaces, not just loopback: a physical iPhone has to reach
    # this over the LAN, and http://localhost means the phone itself.
    cd "$REPO/models/${2:-int8_nc}"
    IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo 127.0.0.1)"
    echo "serving $(pwd)"
    echo "  simulator : http://localhost:8000"
    echo "  device    : http://$IP:8000     <- put this in the app's Model host field"
    python3 -m http.server 8000 --bind 0.0.0.0
    ;;
  *)
    sed -n '1,12p' "$0"
    echo
    echo "usage: tools/dev.sh {xcode|build|test|app|run|serve|push-model} [arg]"
    echo "  xcode                  reset Xcode state and open the demo project"
    echo "  test [debug|release]   run the Swift test suite"
    echo "  app                    build the demo app for the simulator"
    echo "  run                    build, install and launch the demo app"
    echo "  stage [int8_nc]        assemble models/upload/ for hosting"
    echo "  serve [int8_nc]        host the quantized model on :8000"
    ;;
esac
