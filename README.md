# phi-vision

Face detection on the vector unit of an Intel Xeon Phi 7120A (Knights Corner), to take the
detection load of a desktop face tracker off the host CPU.

The detector is YuNet (`face_detection_yunet_2023mar.onnx` from the OpenCV model zoo), run by a small
C interpreter with OpenCV 5.0.0 `FaceDetectorYN`'s pre- and post-processing reproduced from its
source. The same C builds for the host (scalar reference) and for the card, where the convolutions
use KNC vector intrinsics from the Intel-Phi-LLVM clang (`-march=knc`).

## Status (2026-10-23)

| Stage | State |
| --- | --- |
| Host profile of the tracker (OpenCV YuNet, 960x540 input) | Measured: detection 11.2-11.5 ms wall, 31.2-34.1 ms CPU per frame at 30 fps |
| YuNet in C, host | Matches OpenCV on all 12 output heads (max abs difference 1.98e-05) |
| YuNet in C, card, scalar | Same check passes; 99.5 ms per frame on 118 threads |
| YuNet in C, card, vector unit | Same check passes; 27.1-27.5 ms best, 29.3-35.9 ms mean per frame on 118 threads |
| Faces checked against OpenCV | One frame with no faces so far; frames with faces not yet checked |
| Host to card transport, tracker integration | Not started |

Every number comes from `yunet-run` (host or card) or the tracker's own per-second profile; the card
runs used 118 threads, two hardware threads on each of cores 0 to 58.

## Card measurements

`yunet-run yunet.onnx 960 540 118 N frame.bgr`, Xeon Phi 7120A, 61 cores at 1238 MHz:

| Build | Best | Mean | Rounds |
| --- | --- | --- | --- |
| Scalar (x87, knc-cc) | 99.538 ms | 100.346 ms | 10 |
| Vector unit | 27.457 ms | 29.259 ms | 20 |
| Vector unit, sustained | 27.141 ms | 35.883 ms | 1000 |

During the sustained run: the card was 42.6 % busy (phitop), with hardware threads 0 and 1 of cores
0 to 58 fully used, and the core counters recorded 9,115,062,256 VPU instructions in 5 s
(`VPU_INSTRUCTIONS_EXECUTED`, perf event 0x2016).

YuNet at 960x544 (input padded to a multiple of 32) is 0.439 GMAC a frame: pointwise convolutions
68.7 %, depthwise 3x3 18.5 %, the first 3x3 stride-2 convolution 12.9 %.

## Layout

```
lib/onnx.c     ONNX protobuf reader (nodes, float and int64 initializers), no dependencies
lib/pool.c     persistent thread pool
lib/yunet.c    the network: padded NCHW planes, KNC kernels under __KNC__, scalar otherwise;
               OpenCV's padding, decode and NMS
tools/         yunet-run (run and compare), ocv-ref (OpenCV oracle), onnxdump (graph listing)
ref/           OpenCV 5.0.0 sources the pre- and post-processing follow (Apache-2.0)
build.sh       host tools with cc; card binaries with the Intel-Phi-LLVM clang and knc-cc
```

## Building

`bash build.sh` needs `cc`, OpenCV 5 headers and libraries (for `ocv-ref` only), the
[Intel-Phi-LLVM](https://github.com/Lasimeri/Intel-Phi-LLVM) clang (`$PHI_LLVM_BIN`) and the
[Intel-Phi-7120A](https://github.com/Lasimeri/Intel-Phi-7120A) toolchain (`$PHI_TREE`, for knc-cc and
the musl sysroot). The model is not included: `face_detection_yunet_2023mar.onnx` from the
[OpenCV zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), SHA-256
`8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.

## Checking against OpenCV

```
build/ocv-ref MODEL 960 540 frame.bgr REFDIR       # OpenCV's heads and faces for one raw BGR frame
build/yunet-run MODEL 960 540 THREADS ROUNDS frame.bgr REFDIR
```

`yunet-run` prints the largest absolute difference per head and compares the face rows.

## License

MIT for this project's code; see [LICENSE.md](LICENSE.md). `ref/` is OpenCV source, Apache-2.0.
