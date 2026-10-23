/* onnx.h: the parts of an ONNX model a small CNN needs (nodes, float/int64 initializers), read from the protobuf wire
 * format with no library. */
#ifndef ONNX_H
#define ONNX_H

#include <stddef.h>
#include <stdint.h>

#define ONNX_NAME 64

typedef struct {
	char name[ONNX_NAME];
	int ndim;
	int64_t dims[8];
	int type;        /* 1 float, 7 int64 */
	float *f;        /* type 1: n floats */
	int64_t *i;      /* type 7: n int64 */
	size_t n;
} onnx_tensor;

typedef struct {
	char op[24];
	char name[ONNX_NAME];
	int nin, nout;
	char in[8][ONNX_NAME], out[4][ONNX_NAME];
	int group, kernel, stride, pad;   /* Conv / MaxPool (square kernels, symmetric pads) */
	int perm[4], nperm;               /* Transpose */
} onnx_node;

typedef struct {
	int nnode, ninit, noutput;
	onnx_node *node;
	onnx_tensor *init;
	char input[ONNX_NAME];
	char output[16][ONNX_NAME];
} onnx_model;

/* 0, or -1 with a message in err */
int onnx_load(onnx_model *m, const char *path, char *err, size_t errlen);
const onnx_tensor *onnx_init(const onnx_model *m, const char *name);
void onnx_free(onnx_model *m);

#endif
