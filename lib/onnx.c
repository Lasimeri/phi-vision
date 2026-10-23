/* onnx.c: see onnx.h. Field numbers from onnx.proto (ModelProto.graph 7; GraphProto node 1, initializer 5, input 11,
 * output 12; NodeProto input 1, output 2, name 3, op_type 4, attribute 5; AttributeProto name 1, i 3, ints 8;
 * TensorProto dims 1, data_type 2, float_data 4, int64_data 7, name 8, raw_data 9). */
#include "onnx.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const uint8_t *p, *end; } buf;

static uint64_t varint(buf *b)
{
	uint64_t v = 0;
	for (int s = 0; b->p < b->end && s < 64; s += 7) {
		uint8_t c = *b->p++;
		v |= (uint64_t)(c & 0x7f) << s;
		if (!(c & 0x80))
			break;
	}
	return v;
}

static int field(buf *b, int *f, int *wire, buf *sub, uint64_t *val)
{
	if (b->p >= b->end)
		return 0;
	uint64_t key = varint(b);
	*f = (int)(key >> 3);
	*wire = (int)(key & 7);
	switch (*wire) {
	case 0: *val = varint(b); return 1;
	case 1: if (b->end - b->p < 8) return 0; memcpy(val, b->p, 8); b->p += 8; return 1;
	case 5: { uint32_t v; if (b->end - b->p < 4) return 0; memcpy(&v, b->p, 4); *val = v; b->p += 4; return 1; }
	case 2: {
		uint64_t n = varint(b);
		if ((uint64_t)(b->end - b->p) < n) return 0;
		sub->p = b->p; sub->end = b->p + n; b->p += n;
		return 1;
	}
	default: return 0;
	}
}

static void copyname(char *dst, buf s)
{
	size_t n = (size_t)(s.end - s.p);
	if (n >= ONNX_NAME) n = ONNX_NAME - 1;
	memcpy(dst, s.p, n);
	dst[n] = 0;
}

static int tensor(onnx_tensor *t, buf b)
{
	int f, w; buf s; uint64_t v;
	buf raw = { 0, 0 }, fdata = { 0, 0 };
	size_t nf = 0;
	memset(t, 0, sizeof *t);
	while (field(&b, &f, &w, &s, &v)) {
		if (f == 1 && w == 0 && t->ndim < 8) t->dims[t->ndim++] = (int64_t)v;
		else if (f == 1 && w == 2) while (s.p < s.end && t->ndim < 8) t->dims[t->ndim++] = (int64_t)varint(&s);
		else if (f == 2) t->type = (int)v;
		else if (f == 8) copyname(t->name, s);
		else if (f == 9) raw = s;
		else if (f == 4 && w == 2) { fdata = s; nf = (size_t)(s.end - s.p) / 4; }
	}
	size_t n = 1;
	for (int i = 0; i < t->ndim; i++) n *= (size_t)t->dims[i];
	t->n = n;
	if (t->type == 1) {
		t->f = calloc(n ? n : 1, sizeof(float));
		if (!t->f) return -1;
		if (raw.p && (size_t)(raw.end - raw.p) == n * 4) memcpy(t->f, raw.p, n * 4);
		else if (fdata.p && nf == n) memcpy(t->f, fdata.p, n * 4);
	} else if (t->type == 7) {
		t->i = calloc(n ? n : 1, sizeof(int64_t));
		if (!t->i) return -1;
		if (raw.p && (size_t)(raw.end - raw.p) == n * 8) memcpy(t->i, raw.p, n * 8);
	}
	return 0;
}

static void attribute(onnx_node *nd, buf b)
{
	int f, w; buf s; uint64_t v;
	char name[32] = { 0 };
	int64_t iv = 0, ints[8]; int nints = 0;
	while (field(&b, &f, &w, &s, &v)) {
		if (f == 1) { size_t n = (size_t)(s.end - s.p); if (n > 31) n = 31; memcpy(name, s.p, n); name[n] = 0; }
		else if (f == 3) iv = (int64_t)v;
		else if (f == 8 && w == 2) while (s.p < s.end && nints < 8) ints[nints++] = (int64_t)varint(&s);
		else if (f == 8 && w == 0 && nints < 8) ints[nints++] = (int64_t)v;
	}
	if (!strcmp(name, "group")) nd->group = (int)iv;
	else if (!strcmp(name, "kernel_shape") && nints) nd->kernel = (int)ints[0];
	else if (!strcmp(name, "strides") && nints) nd->stride = (int)ints[0];
	else if (!strcmp(name, "pads") && nints) nd->pad = (int)ints[0];
	else if (!strcmp(name, "perm")) { nd->nperm = nints > 4 ? 4 : nints; for (int i = 0; i < nd->nperm; i++) nd->perm[i] = (int)ints[i]; }
}

static void node(onnx_node *nd, buf b)
{
	int f, w; buf s; uint64_t v;
	memset(nd, 0, sizeof *nd);
	nd->group = 1; nd->kernel = 1; nd->stride = 1;
	while (field(&b, &f, &w, &s, &v)) {
		if (f == 1 && nd->nin < 8) copyname(nd->in[nd->nin++], s);
		else if (f == 2 && nd->nout < 4) copyname(nd->out[nd->nout++], s);
		else if (f == 3) copyname(nd->name, s);
		else if (f == 4) { size_t n = (size_t)(s.end - s.p); if (n > 23) n = 23; memcpy(nd->op, s.p, n); nd->op[n] = 0; }
		else if (f == 5) attribute(nd, s);
	}
}

static void valuename(char *dst, buf vi)
{
	int f, w; buf s; uint64_t v;
	while (field(&vi, &f, &w, &s, &v)) if (f == 1) copyname(dst, s);
}

int onnx_load(onnx_model *m, const char *path, char *err, size_t errlen)
{
	memset(m, 0, sizeof *m);
	FILE *fp = fopen(path, "rb");
	if (!fp) { snprintf(err, errlen, "%s: cannot open", path); return -1; }
	fseek(fp, 0, SEEK_END);
	long len = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	uint8_t *data = malloc(len > 0 ? (size_t)len : 1);
	if (!data || fread(data, 1, (size_t)len, fp) != (size_t)len) { fclose(fp); free(data); snprintf(err, errlen, "%s: read failed", path); return -1; }
	fclose(fp);

	buf mb = { data, data + len }, s, g = { 0, 0 };
	int f, w; uint64_t v;
	while (field(&mb, &f, &w, &s, &v)) if (f == 7) g = s;
	if (!g.p) { free(data); snprintf(err, errlen, "%s: no graph", path); return -1; }

	buf gb = g;
	while (field(&gb, &f, &w, &s, &v)) {
		if (f == 1) m->nnode++;
		else if (f == 5) m->ninit++;
	}
	m->node = calloc((size_t)m->nnode, sizeof *m->node);
	m->init = calloc((size_t)m->ninit, sizeof *m->init);
	if (!m->node || !m->init) { free(data); snprintf(err, errlen, "out of memory"); return -1; }
	int in = 0, ii = 0, gotinput = 0;
	gb = g;
	while (field(&gb, &f, &w, &s, &v)) {
		if (f == 1) node(&m->node[in++], s);
		else if (f == 5 && tensor(&m->init[ii++], s)) { free(data); snprintf(err, errlen, "out of memory"); return -1; }
		else if (f == 11 && !gotinput) { valuename(m->input, s); gotinput = 1; }
		else if (f == 12 && m->noutput < 16) valuename(m->output[m->noutput++], s);
	}
	free(data);
	return 0;
}

const onnx_tensor *onnx_init(const onnx_model *m, const char *name)
{
	for (int i = 0; i < m->ninit; i++)
		if (!strcmp(m->init[i].name, name))
			return &m->init[i];
	return NULL;
}

void onnx_free(onnx_model *m)
{
	for (int i = 0; i < m->ninit; i++) { free(m->init[i].f); free(m->init[i].i); }
	free(m->init);
	free(m->node);
	memset(m, 0, sizeof *m);
}
