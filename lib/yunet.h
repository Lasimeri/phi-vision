/* yunet.h: YuNet face detection (face_detection_yunet_2023mar.onnx) in C, with OpenCV 5.0.0 FaceDetectorYN's pre- and
 * post-processing: the BGR picture zero-padded to a multiple of 32, fed as planar floats 0..255; per stride 8/16/32
 * score = sqrt(clamp(cls) * clamp(obj)), box and landmark decode, then NMS on integer-truncated boxes. */
#ifndef YUNET_H
#define YUNET_H

#include <stddef.h>
#include <stdint.h>

#include "pool.h"

#define YUNET_FACE 15   /* x, y, w, h, 5 landmarks (x, y), score: as cv::FaceDetectorYN rows */

typedef struct yunet yunet;

/* w x h is the picture size; the pool (may be NULL) runs the layers */
yunet *yunet_open(const char *onnx, int w, int h, pool *pl, char *err, size_t errlen);

/* runs the network on a w x h BGR picture (rows stride bytes apart); returns the number of faces written to out
 * (at most maxf rows of YUNET_FACE floats), best first as OpenCV returns them */
int yunet_detect(yunet *y, const uint8_t *bgr, int stride, float score, float nms, int top_k, float *out, int maxf);

/* after yunet_detect: a named tensor (the 12 heads cls_8 ... kps_32, or any node output) and its element count */
const float *yunet_tensor(yunet *y, const char *name, size_t *n);

/* per-layer seconds of the last yunet_detect, summed by op ("Conv", "Relu", ...); returns the count written */
int yunet_timing(yunet *y, char names[][24], double *secs, int max);

void yunet_close(yunet *y);

#endif
