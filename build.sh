#!/usr/bin/env bash
# build.sh: host tools (cc) and card binaries into build/. On the card: lib/yunet.c through the Intel-Phi-LLVM S3 clang
# (-march=knc: the convolutions on the vector unit), the rest through the stack's knc-cc (scalar x87, static).
# yunet-run-card-scalar is the same program with the scalar kernels, the card-side reference.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out="$here/build"
mkdir -p "$out"
flags=(-O2 -std=gnu11 -Wall -Wextra)
lib=("$here/lib/onnx.c" "$here/lib/pool.c")
cc "${flags[@]}" -o "$out/onnxdump" "$here/tools/onnxdump.c"
cc "${flags[@]}" -o "$out/yunet-run" "$here/tools/yunet-run.c" "$here/lib/yunet.c" "${lib[@]}" -lm -lpthread
cc "${flags[@]}" -o "$out/corpus-gate" "$here/tools/corpus-gate.c" "$here/lib/yunet.c" "${lib[@]}" -lm -lpthread
cc "${flags[@]}" -o "$out/pvtest" "$here/tools/pvtest.c" "$here/host/phiv.c" -lm
cc "${flags[@]}" -fPIC -c -o "$out/phiv.o" "$here/host/phiv.c"
cv=/usr/include/opencv5
g++ -O2 -o "$out/ocv-ref" "$here/tools/ocv-ref.cpp" -I"$cv" -lopencv_core -lopencv_imgproc -lopencv_dnn -lopencv_objdetect \
    -lopencv_flann -lopencv_features -lopencv_calib -lopencv_geometry
VCC="${PHI_LLVM_BIN:-$HOME/.cache/intel-phi-llvm-build/llvm/bin}/clang"
SYS="${PHI_SYSROOT_7120A:-$HOME/.cache/intel-phi-7120a-build/toolchain/sysroot}"
"$VCC" --target=x86_64-unknown-linux-musl -march=knc --sysroot="$SYS" "${flags[@]}" -fno-jump-tables -falign-loops=1 -fPIE \
    -c -o "$out/yunet-knc.o" "$here/lib/yunet.c"
# shellcheck disable=SC1091
. "${PHI_TREE:-$HOME/Intel Phi 7120A}/toolchain/env.sh"
knc-cc "${flags[@]}" -static -o "$out/pvd-card" "$here/card/pvd.c" "$here/card/cdma.c" "$out/yunet-knc.o" "${lib[@]}" -lm -lpthread
knc-cc "${flags[@]}" -static -o "$out/yunet-run-card" "$here/tools/yunet-run.c" "$out/yunet-knc.o" "${lib[@]}" -lm -lpthread
knc-cc "${flags[@]}" -static -o "$out/corpus-gate-card" "$here/tools/corpus-gate.c" "$out/yunet-knc.o" "${lib[@]}" -lm -lpthread
knc-cc "${flags[@]}" -static -o "$out/yunet-run-card-scalar" "$here/tools/yunet-run.c" "$here/lib/yunet.c" "${lib[@]}" -lm -lpthread
audit="${PHI_TREE:-$HOME/Intel Phi 7120A}/host/target/debug/phi-isa-audit"
for b in pvd-card yunet-run-card corpus-gate-card yunet-run-card-scalar; do printf '%s: ' "$b"; "$audit" "$out/$b" | tail -1; done
