/* yunet.c: see yunet.h. A small interpreter over the model's nodes (Conv, Relu, MaxPool, Resize, Add, Transpose,
 * Reshape, Sigmoid), shapes fixed at open for the padded picture size. A Relu that is a Conv's only consumer is folded
 * into it.
 *
 * Layout: every NCHW feature map keeps each channel plane with zero margins: one zero row above and below, 16 zero
 * floats left of each row and at least one right of it, rows a multiple of 16 floats (64 bytes) apart. A 3x3 kernel
 * with pad 1 then reads its neighbours without bounds checks, and on the card each row starts on a vector boundary.
 * Writers never touch the margins (the card's kernels mask the last vector of a row). The input picture becomes two
 * planes per colour, even and odd columns (the odd plane shifted one right), so the first convolution's stride 2 reads
 * contiguous columns. Head tensors after Transpose are flat, as OpenCV's.
 *
 * Built for the card (clang -march=knc) the convolutions run on the vector unit; built anywhere else the same plan
 * runs in scalar C (the host reference). */
#include "yunet.h"
#include "onnx.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __KNC__
#include <immintrin.h>
#endif

typedef struct {
	char name[ONNX_NAME];
	int64_t d[4];          /* N, C, H, W (flat head tensors: 1, H*W, k, 1) */
	size_t n;              /* logical elements */
	int padded;            /* the margin layout above */
	int rs;                /* row stride, floats */
	size_t ps;             /* plane stride */
	size_t off;            /* element (0, 0) of a plane */
	float *data;
	int own;
} tens;

#define PLANE(t, c) ((t)->data + (size_t)(c) * (t)->ps + (t)->off)   /* element (c, 0, 0) */

enum { OP_CONV, OP_RELU, OP_MAXPOOL, OP_RESIZE, OP_ADD, OP_TRANSPOSE, OP_RESHAPE, OP_SIGMOID, OP_NONE };
static const char *opname[] = { "Conv", "Relu", "MaxPool", "Resize", "Add", "Transpose", "Reshape", "Sigmoid", "none" };
enum { T_INPUT = OP_NONE, T_DECODE, T_N };   /* extra timing slots */

typedef struct {
	int op;
	int in[2], out;        /* tensor indexes */
	const float *w, *b;    /* Conv */
	int group, k, s, p, relu, first;
	int rows;              /* output rows per task */
	int ocb;               /* Conv: output channels per task */
} step;

struct yunet {
	onnx_model m;
	pool *pl;
	int threads;
	int w, h, pw, ph;
	tens *t;
	int nt;
	step *st;
	int nst;
	int input;             /* the network input: its even / odd column planes */
	int even, odd;
	const uint8_t *bgr;    /* yunet_detect's picture, for the input tasks */
	int bgr_stride;
	double optime[T_N];
};

static int tindex(yunet *y, const char *name)
{
	for (int i = 0; i < y->nt; i++)
		if (!strcmp(y->t[i].name, name))
			return i;
	return -1;
}

static int tnew(yunet *y, const char *name, int64_t n0, int64_t n1, int64_t n2, int64_t n3, int padded)
{
	tens *t = &y->t[y->nt];
	memset(t, 0, sizeof *t);
	snprintf(t->name, sizeof t->name, "%s", name);
	t->d[0] = n0; t->d[1] = n1; t->d[2] = n2; t->d[3] = n3;
	t->n = (size_t)(n0 * n1 * n2 * n3);
	t->padded = padded;
	if (padded) {
		t->rs = (int)((16 + n3 + 1 + 15) / 16 * 16);
		t->ps = (size_t)t->rs * (size_t)(n2 + 2);
		t->off = (size_t)t->rs + 16;
	} else {
		t->rs = (int)n3;
		t->ps = (size_t)(n2 * n3);
		t->off = 0;
	}
	return y->nt++;
}

static int talloc(tens *t)
{
	size_t n = (t->padded ? (size_t)t->d[1] * t->ps : t->n) + 64;
	void *p = NULL;
	if (posix_memalign(&p, 64, n * sizeof(float))) return -1;
	memset(p, 0, n * sizeof(float));
	t->data = p;
	t->own = 1;
	return 0;
}

static double now(void)
{
	struct timespec s;
	clock_gettime(CLOCK_MONOTONIC, &s);
	return s.tv_sec + s.tv_nsec / 1e9;
}

/* ---- tasks ---- */

typedef struct { yunet *y; const step *s; } job;

/* task t of a (channels x row blocks) split: the channel block and the rows */
static void split(const step *s, int oh, int task, int *cblk, int *y0, int *y1)
{
	const int nblk = (oh + s->rows - 1) / s->rows;
	*cblk = task / nblk;
	*y0 = (task % nblk) * s->rows;
	*y1 = *y0 + s->rows < oh ? *y0 + s->rows : oh;
}

/* the picture into the even and odd column planes: one task per input row */
static void input_task(void *v, int r)
{
	const job *j = v;
	yunet *y = j->y;
	const tens *E = &y->t[y->even], *O = &y->t[y->odd];
	const uint8_t *src = y->bgr + (size_t)r * y->bgr_stride;
	for (int c = 0; c < 3; c++) {
		float *e = PLANE(E, c) + (size_t)r * E->rs, *o = PLANE(O, c) + (size_t)r * O->rs;
		for (int x = 0; x < y->w; x += 2) e[x / 2] = src[3 * x + c];
		for (int x = 1; x < y->w; x += 2) o[(x + 1) / 2] = src[3 * x + c];   /* odd column 2j-1 -> o[j] */
	}
}

#ifdef __KNC__

static inline __m512 loadu(const float *p)
{
	__m512 v = _mm512_undefined_ps();
	v = _mm512_loadunpacklo_ps(v, p);
	return _mm512_loadunpackhi_ps(v, p + 16);
}

static inline __m512 relu16(__m512 v)
{
	return _mm512_castsi512_ps(_mm512_max_epi32(_mm512_castps_si512(v), _mm512_setzero_epi32()));
}

static inline __mmask16 tailmask(int n)   /* the first n lanes (n >= 16: all) */
{
	return _mm512_int2mask(n >= 16 ? 0xffff : (1 << n) - 1);
}

/* pointwise, 8 output channels x 32 columns per inner block */
static void conv_pw8(const step *s, const tens *I, const tens *O, int oc0, int y0, int y1)
{
	const int cin = (int)I->d[1], ow = (int)O->d[3];
	const float *w = s->w + (size_t)oc0 * cin;
	for (int yy = y0; yy < y1; yy++) {
		for (int x = 0; x < ow; x += 32) {
			const int two = x + 16 < ow;
			__m512 a[8][2];
			for (int j = 0; j < 8; j++) a[j][0] = a[j][1] = _mm512_set1_ps(s->b ? s->b[oc0 + j] : 0.f);
			const float *src = PLANE(I, 0) + (size_t)yy * I->rs + x;
			for (int ic = 0; ic < cin; ic++, src += I->ps) {
				const __m512 i0 = _mm512_load_ps(src), i1 = _mm512_load_ps(src + 16);
				for (int j = 0; j < 8; j++) {
					const __m512 wv = _mm512_set1_ps(w[(size_t)j * cin + ic]);
					a[j][0] = _mm512_fmadd_ps(i0, wv, a[j][0]);
					a[j][1] = _mm512_fmadd_ps(i1, wv, a[j][1]);
				}
			}
			for (int j = 0; j < 8; j++) {
				float *o = PLANE(O, oc0 + j) + (size_t)yy * O->rs + x;
				__m512 r0 = s->relu ? relu16(a[j][0]) : a[j][0], r1 = s->relu ? relu16(a[j][1]) : a[j][1];
				_mm512_mask_store_ps(o, tailmask(ow - x), r0);
				if (two) _mm512_mask_store_ps(o + 16, tailmask(ow - x - 16), r1);
			}
		}
	}
}

/* pointwise, one output channel (the heads: 1, 4, 10 channels) */
static void conv_pw1(const step *s, const tens *I, const tens *O, int oc, int y0, int y1)
{
	const int cin = (int)I->d[1], ow = (int)O->d[3];
	const float *w = s->w + (size_t)oc * cin;
	for (int yy = y0; yy < y1; yy++)
		for (int x = 0; x < ow; x += 16) {
			__m512 a = _mm512_set1_ps(s->b ? s->b[oc] : 0.f);
			const float *src = PLANE(I, 0) + (size_t)yy * I->rs + x;
			for (int ic = 0; ic < cin; ic++, src += I->ps) a = _mm512_fmadd_ps(_mm512_load_ps(src), _mm512_set1_ps(w[ic]), a);
			_mm512_mask_store_ps(PLANE(O, oc) + (size_t)yy * O->rs + x, tailmask(ow - x), s->relu ? relu16(a) : a);
		}
}

/* depthwise 3x3, stride 1, pad 1 (the margins are the padding) */
static void conv_dw(const step *s, const tens *I, const tens *O, int c, int y0, int y1)
{
	const int ow = (int)O->d[3];
	const float *wt = s->w + (size_t)c * 9;
	__m512 w[9];
	for (int k = 0; k < 9; k++) w[k] = _mm512_set1_ps(wt[k]);
	const __m512 b = _mm512_set1_ps(s->b ? s->b[c] : 0.f);
	const float *pl = PLANE(I, c);
	for (int yy = y0; yy < y1; yy++) {
		const float *r0 = pl + (size_t)(yy - 1) * I->rs, *r1 = r0 + I->rs, *r2 = r1 + I->rs;
		float *o = PLANE(O, c) + (size_t)yy * O->rs;
		for (int x = 0; x < ow; x += 16) {
			__m512 a = b;
			a = _mm512_fmadd_ps(loadu(r0 + x - 1), w[0], a);
			a = _mm512_fmadd_ps(_mm512_load_ps(r0 + x), w[1], a);
			a = _mm512_fmadd_ps(loadu(r0 + x + 1), w[2], a);
			a = _mm512_fmadd_ps(loadu(r1 + x - 1), w[3], a);
			a = _mm512_fmadd_ps(_mm512_load_ps(r1 + x), w[4], a);
			a = _mm512_fmadd_ps(loadu(r1 + x + 1), w[5], a);
			a = _mm512_fmadd_ps(loadu(r2 + x - 1), w[6], a);
			a = _mm512_fmadd_ps(_mm512_load_ps(r2 + x), w[7], a);
			a = _mm512_fmadd_ps(loadu(r2 + x + 1), w[8], a);
			_mm512_mask_store_ps(o + x, tailmask(ow - x), s->relu ? relu16(a) : a);
		}
	}
}

/* the first layer: 3x3, stride 2, 3 -> 16 channels, from the even / odd column planes, all 16 outputs at once */
static void conv_first(const yunet *y, const step *s, const tens *O, int y0, int y1)
{
	const tens *E = &y->t[y->even], *Q = &y->t[y->odd];
	const int ow = (int)O->d[3];
	for (int yy = y0; yy < y1; yy++)
		for (int x = 0; x < ow; x += 16) {
			__m512 a[16];
			for (int oc = 0; oc < 16; oc++) a[oc] = _mm512_set1_ps(s->b ? s->b[oc] : 0.f);
			for (int c = 0; c < 3; c++)
				for (int ky = 0; ky < 3; ky++) {
					const int iy = 2 * yy - 1 + ky;
					const float *e = PLANE(E, c) + (ptrdiff_t)iy * E->rs + x, *q = PLANE(Q, c) + (ptrdiff_t)iy * Q->rs + x;
					const __m512 q0 = _mm512_load_ps(q), e1 = _mm512_load_ps(e), q2 = loadu(q + 1);
					const float *wk = s->w + (size_t)(c * 3 + ky) * 3;
					for (int oc = 0; oc < 16; oc++, wk += 27) {
						a[oc] = _mm512_fmadd_ps(q0, _mm512_set1_ps(wk[0]), a[oc]);
						a[oc] = _mm512_fmadd_ps(e1, _mm512_set1_ps(wk[1]), a[oc]);
						a[oc] = _mm512_fmadd_ps(q2, _mm512_set1_ps(wk[2]), a[oc]);
					}
				}
			for (int oc = 0; oc < 16; oc++)
				_mm512_mask_store_ps(PLANE(O, oc) + (size_t)yy * O->rs + x, tailmask(ow - x), s->relu ? relu16(a[oc]) : a[oc]);
		}
}

static void conv_task(void *v, int task)
{
	const job *j = v;
	const step *s = j->s;
	const tens *I = &j->y->t[s->in[0]], *O = &j->y->t[s->out];
	int cb, y0, y1;
	split(s, (int)O->d[2], task, &cb, &y0, &y1);
	if (s->first) conv_first(j->y, s, O, y0, y1);
	else if (s->k == 1 && s->ocb == 8) conv_pw8(s, I, O, cb * 8, y0, y1);
	else if (s->k == 1) conv_pw1(s, I, O, cb, y0, y1);
	else conv_dw(s, I, O, cb, y0, y1);
}

#else /* scalar */

static void conv_task(void *v, int task)
{
	const job *j = v;
	const step *s = j->s;
	const yunet *y = j->y;
	const tens *I = &y->t[s->in[0]], *O = &y->t[s->out];
	const int cin = (int)I->d[1], cout = (int)O->d[1], ow = (int)O->d[3];
	const int k = s->k, st = s->s, p = s->p, cig = cin / s->group, cog = cout / s->group;
	int cb, y0, y1;
	split(s, (int)O->d[2], task, &cb, &y0, &y1);
	for (int oc = cb * s->ocb; oc < (cb + 1) * s->ocb && oc < cout; oc++) {
		const int g = oc / cog;
		const float *wt = s->w + (size_t)oc * cig * k * k;
		const float bias = s->b ? s->b[oc] : 0.f;
		for (int yy = y0; yy < y1; yy++) {
			float *o = PLANE(O, oc) + (size_t)yy * O->rs;
			for (int x = 0; x < ow; x++) o[x] = bias;
			for (int ic = 0; ic < cig; ic++) {
				if (s->first) {   /* from the even / odd column planes: column 2x-1+kx */
					const tens *E = &y->t[y->even], *Q = &y->t[y->odd];
					for (int ky = 0; ky < 3; ky++) {
						const int iy = 2 * yy - 1 + ky;
						const float *e = PLANE(E, ic) + (ptrdiff_t)iy * E->rs, *q = PLANE(Q, ic) + (ptrdiff_t)iy * Q->rs;
						const float w0 = wt[(ic * 3 + ky) * 3], w1 = wt[(ic * 3 + ky) * 3 + 1], w2 = wt[(ic * 3 + ky) * 3 + 2];
						for (int x = 0; x < ow; x++) o[x] += w0 * q[x] + w1 * e[x] + w2 * q[x + 1];
					}
					continue;
				}
				const float *src = PLANE(I, g * cig + ic);
				for (int ky = 0; ky < k; ky++) {
					const float *row = src + (ptrdiff_t)(yy * st - p + ky) * I->rs - p;
					for (int kx = 0; kx < k; kx++) {
						const float wv = wt[(ic * k + ky) * k + kx];
						const float *r = row + kx;
						if (st == 1) for (int x = 0; x < ow; x++) o[x] += wv * r[x];
						else for (int x = 0; x < ow; x++) o[x] += wv * r[x * st];
					}
				}
			}
			if (s->relu) for (int x = 0; x < ow; x++) if (o[x] < 0.f) o[x] = 0.f;
		}
	}
}

#endif

static void relu_task(void *v, int task)
{
	const job *j = v;
	const tens *I = &j->y->t[j->s->in[0]], *O = &j->y->t[j->s->out];
	int c, y0, y1;
	split(j->s, (int)O->d[2], task, &c, &y0, &y1);
	for (int yy = y0; yy < y1; yy++) {
		const float *a = PLANE(I, c) + (size_t)yy * I->rs;
		float *o = PLANE(O, c) + (size_t)yy * O->rs;
		for (int x = 0; x < (int)O->d[3]; x++) o[x] = a[x] < 0.f ? 0.f : a[x];
	}
}

static void maxpool_task(void *v, int task)
{
	const job *j = v;
	const tens *I = &j->y->t[j->s->in[0]], *O = &j->y->t[j->s->out];
	int c, y0, y1;
	split(j->s, (int)O->d[2], task, &c, &y0, &y1);
	const int ow = (int)O->d[3];
	for (int yy = y0; yy < y1; yy++) {
		const float *r0 = PLANE(I, c) + (size_t)(2 * yy) * I->rs, *r1 = r0 + I->rs;
		float *o = PLANE(O, c) + (size_t)yy * O->rs;
		for (int x = 0; x < ow; x++) {
			float m = r0[2 * x];
			if (r0[2 * x + 1] > m) m = r0[2 * x + 1];
			if (r1[2 * x] > m) m = r1[2 * x];
			if (r1[2 * x + 1] > m) m = r1[2 * x + 1];
			o[x] = m;
		}
	}
}

static void resize_task(void *v, int task)   /* nearest, asymmetric, floor, integer scale */
{
	const job *j = v;
	const tens *I = &j->y->t[j->s->in[0]], *O = &j->y->t[j->s->out];
	int c, y0, y1;
	split(j->s, (int)O->d[2], task, &c, &y0, &y1);
	const int sy = (int)(O->d[2] / I->d[2]), sx = (int)(O->d[3] / I->d[3]), ow = (int)O->d[3];
	for (int yy = y0; yy < y1; yy++) {
		const float *a = PLANE(I, c) + (size_t)(yy / sy) * I->rs;
		float *o = PLANE(O, c) + (size_t)yy * O->rs;
		for (int x = 0; x < ow; x++) o[x] = a[x / sx];
	}
}

static void add_task(void *v, int task)
{
	const job *j = v;
	const tens *A = &j->y->t[j->s->in[0]], *B = &j->y->t[j->s->in[1]], *O = &j->y->t[j->s->out];
	int c, y0, y1;
	split(j->s, (int)O->d[2], task, &c, &y0, &y1);
	for (int yy = y0; yy < y1; yy++) {
		const float *a = PLANE(A, c) + (size_t)yy * A->rs, *b = PLANE(B, c) + (size_t)yy * B->rs;
		float *o = PLANE(O, c) + (size_t)yy * O->rs;
		for (int x = 0; x < (int)O->d[3]; x++) o[x] = a[x] + b[x];
	}
}

static void transpose_task(void *v, int c)   /* padded [1, C, H, W] -> flat [1, H, W, C], one channel per task */
{
	const job *j = v;
	const tens *I = &j->y->t[j->s->in[0]], *O = &j->y->t[j->s->out];
	const int nc = (int)I->d[1], ih = (int)I->d[2], iw = (int)I->d[3];
	for (int yy = 0; yy < ih; yy++) {
		const float *a = PLANE(I, c) + (size_t)yy * I->rs;
		for (int x = 0; x < iw; x++) O->data[((size_t)yy * iw + x) * nc + c] = a[x];
	}
}

static void sigmoid_task(void *v, int task)
{
	const job *j = v;
	const tens *I = &j->y->t[j->s->in[0]], *O = &j->y->t[j->s->out];
	const size_t per = (I->n + 63) / 64, i0 = (size_t)task * per, i1 = i0 + per < I->n ? i0 + per : I->n;
	for (size_t i = i0; i < i1; i++) O->data[i] = 1.f / (1.f + expf(-I->data[i]));
}

static void run(yunet *y, pool_fn fn, const step *s, int ntasks)
{
	job jb = { y, s };
	if (y->pl) pool_run(y->pl, fn, &jb, ntasks);
	else for (int t = 0; t < ntasks; t++) fn(&jb, t);
}

/* rows per task so a layer of `blocks` channel blocks makes about 4 tasks a thread */
static int rows_for(int threads, int blocks, int oh)
{
	int want = threads * 4 / (blocks > 0 ? blocks : 1);
	if (want < 1) want = 1;
	if (want > oh) want = oh;
	return (oh + want - 1) / want;
}

/* ---- open ---- */

yunet *yunet_open(const char *path, int w, int h, pool *pl, char *err, size_t errlen)
{
	yunet *y = calloc(1, sizeof *y);
	if (!y) { snprintf(err, errlen, "out of memory"); return NULL; }
	if (onnx_load(&y->m, path, err, errlen)) { free(y); return NULL; }
	y->pl = pl;
	y->threads = pl ? pool_threads(pl) : 1;
	y->w = w; y->h = h;
	y->pw = ((w - 1) / 32 + 1) * 32;
	y->ph = ((h - 1) / 32 + 1) * 32;
	y->t = calloc((size_t)y->m.nnode * 2 + 8, sizeof *y->t);
	y->st = calloc((size_t)y->m.nnode, sizeof *y->st);
	if (!y->t || !y->st) { snprintf(err, errlen, "out of memory"); yunet_close(y); return NULL; }
	y->input = tnew(y, y->m.input, 1, 3, y->ph, y->pw, 1);   /* shape only: the planes below hold it */
	y->even = tnew(y, "(even columns)", 1, 3, y->ph, y->pw / 2, 1);
	y->odd = tnew(y, "(odd columns)", 1, 3, y->ph, y->pw / 2 + 1, 1);
	if (talloc(&y->t[y->even]) || talloc(&y->t[y->odd])) { snprintf(err, errlen, "out of memory"); yunet_close(y); return NULL; }

	for (int i = 0; i < y->m.nnode; i++) {
		const onnx_node *nd = &y->m.node[i];
		step *s = &y->st[y->nst];
		memset(s, 0, sizeof *s);
		s->op = OP_NONE;
		for (int o = 0; o < OP_NONE; o++) if (!strcmp(nd->op, opname[o])) s->op = o;
		if (s->op == OP_NONE) { snprintf(err, errlen, "unsupported op %s (%s)", nd->op, nd->name); yunet_close(y); return NULL; }
		s->in[0] = tindex(y, nd->in[0]);
		if (s->in[0] < 0) { snprintf(err, errlen, "%s: input %s unknown", nd->name, nd->in[0]); yunet_close(y); return NULL; }
		const tens *I = &y->t[s->in[0]];
		int64_t n = I->d[0], c = I->d[1], hh = I->d[2], ww = I->d[3];
		int alias = 0, padded = 1, blocks = (int)c;
		switch (s->op) {
		case OP_CONV: {
			const onnx_tensor *wt = onnx_init(&y->m, nd->in[1]), *bt = nd->nin > 2 ? onnx_init(&y->m, nd->in[2]) : NULL;
			if (!wt || wt->ndim != 4) { snprintf(err, errlen, "%s: weights", nd->name); yunet_close(y); return NULL; }
			s->w = wt->f; s->b = bt ? bt->f : NULL;
			s->group = nd->group; s->k = nd->kernel; s->s = nd->stride; s->p = nd->pad;
			c = wt->dims[0];
			hh = (hh + 2 * s->p - s->k) / s->s + 1;
			ww = (ww + 2 * s->p - s->k) / s->s + 1;
			s->first = s->in[0] == y->input;
			const int dw = s->group == I->d[1] && s->group == c;
			if (s->first ? !(s->k == 3 && s->s == 2 && s->p == 1 && I->d[1] == 3 && c == 16)
			             : !((s->k == 1 && s->s == 1 && s->p == 0 && s->group == 1) || (dw && s->k == 3 && s->s == 1 && s->p == 1))) {
				snprintf(err, errlen, "%s: unsupported convolution (k %d s %d p %d group %d)", nd->name, s->k, s->s, s->p, s->group);
				yunet_close(y); return NULL;
			}
			s->ocb = s->first ? 16 : (s->k == 1 && c % 8 == 0) ? 8 : 1;
			blocks = (int)(c / s->ocb);
			/* fold the next node's Relu when it is this output's only consumer */
			if (i + 1 < y->m.nnode && !strcmp(y->m.node[i + 1].op, "Relu") && !strcmp(y->m.node[i + 1].in[0], nd->out[0])) {
				int users = 0;
				for (int q = 0; q < y->m.nnode; q++)
					for (int a = 0; a < y->m.node[q].nin; a++) users += !strcmp(y->m.node[q].in[a], nd->out[0]);
				if (users == 1) s->relu = 1;
			}
			break;
		}
		case OP_RELU:
			if (y->nst > 0 && y->st[y->nst - 1].op == OP_CONV && y->st[y->nst - 1].relu && y->st[y->nst - 1].out == s->in[0]) alias = 1;
			break;
		case OP_MAXPOOL: hh /= 2; ww /= 2; break;
		case OP_RESIZE: {
			const onnx_tensor *sc = nd->nin > 2 ? onnx_init(&y->m, nd->in[2]) : NULL;
			int fy = sc && sc->n == 4 ? (int)sc->f[2] : 2, fx = sc && sc->n == 4 ? (int)sc->f[3] : 2;
			hh *= fy; ww *= fx;
			break;
		}
		case OP_ADD:
			s->in[1] = tindex(y, nd->in[1]);
			if (s->in[1] < 0 || y->t[s->in[1]].n != I->n) { snprintf(err, errlen, "%s: add shapes", nd->name); yunet_close(y); return NULL; }
			break;
		case OP_TRANSPOSE:
			if (nd->nperm != 4 || nd->perm[0] != 0 || nd->perm[1] != 2 || nd->perm[2] != 3 || nd->perm[3] != 1) {
				snprintf(err, errlen, "%s: only perm 0,2,3,1", nd->name); yunet_close(y); return NULL;
			}
			{ int64_t cc = c; c = hh; hh = ww; ww = cc; }   /* [1, H, W, C] */
			padded = 0;
			break;
		case OP_RESHAPE: {   /* [1, H, W, C] -> [1, H*W, C]: same data */
			int64_t last = I->d[3];
			n = 1; c = (int64_t)I->n / last; hh = last; ww = 1;
			alias = 1; padded = 0;
			break;
		}
		case OP_SIGMOID: padded = 0; break;
		}
		s->out = tnew(y, nd->out[0], n, c, hh, ww, padded);
		if (padded) s->rows = rows_for(y->threads, blocks, (int)hh);
		if (alias) { y->t[s->out].data = y->t[s->in[0]].data; s->op = OP_NONE; }
		else if (talloc(&y->t[s->out])) { snprintf(err, errlen, "out of memory"); yunet_close(y); return NULL; }
		y->nst++;
	}
	return y;
}

/* ---- detect ---- */

static void forward(yunet *y)
{
	for (int i = 0; i < y->nst; i++) {
		const step *s = &y->st[i];
		const tens *I = &y->t[s->in[0]], *O = &y->t[s->out];
		const int rowtasks = s->rows ? (int)((O->d[2] + s->rows - 1) / s->rows) : 1;
		double t0 = now();
		switch (s->op) {
		case OP_CONV: run(y, conv_task, s, (int)(O->d[1] / s->ocb) * rowtasks); break;
		case OP_RELU: run(y, relu_task, s, (int)O->d[1] * rowtasks); break;
		case OP_MAXPOOL: run(y, maxpool_task, s, (int)O->d[1] * rowtasks); break;
		case OP_RESIZE: run(y, resize_task, s, (int)O->d[1] * rowtasks); break;
		case OP_ADD: run(y, add_task, s, (int)O->d[1] * rowtasks); break;
		case OP_TRANSPOSE: run(y, transpose_task, s, (int)I->d[1]); break;
		case OP_SIGMOID: run(y, sigmoid_task, s, 64); break;
		default: break;
		}
		if (s->op != OP_NONE) y->optime[s->op] += now() - t0;
	}
}

typedef struct { float score; int idx; } scored;

static int cmp_desc(const void *a, const void *b)   /* score descending, then index (std::stable_sort order) */
{
	const scored *x = a, *z = b;
	if (x->score > z->score) return -1;
	if (x->score < z->score) return 1;
	return x->idx - z->idx;
}

/* rectOverlap on cv::Rect (int): 1.f - (float)jaccardDistance, with its "both areas zero -> distance 0" rule */
static float overlap(const int *a, const int *b)
{
	int aa = a[2] * a[3], ab = b[2] * b[3];
	if (aa + ab <= 0) return 1.f;
	int x0 = a[0] > b[0] ? a[0] : b[0], y0 = a[1] > b[1] ? a[1] : b[1];
	int x1 = a[0] + a[2] < b[0] + b[2] ? a[0] + a[2] : b[0] + b[2];
	int y1 = a[1] + a[3] < b[1] + b[3] ? a[1] + a[3] : b[1] + b[3];
	double inter = (x1 > x0 && y1 > y0) ? (double)(x1 - x0) * (y1 - y0) : 0;
	return 1.f - (float)(1.0 - inter / (aa + ab - inter));
}

int yunet_detect(yunet *y, const uint8_t *bgr, int stride, float score_thr, float nms_thr, int top_k, float *out, int maxf)
{
	memset(y->optime, 0, sizeof y->optime);
	double t0 = now();
	y->bgr = bgr;
	y->bgr_stride = stride;
	{ job jb = { y, NULL }; if (y->pl) pool_run(y->pl, input_task, &jb, y->h); else for (int r = 0; r < y->h; r++) input_task(&jb, r); }
	y->optime[T_INPUT] = now() - t0;
	forward(y);
	t0 = now();

	/* decode, as FaceDetectorYNImpl::postProcess */
	static const int strides[3] = { 8, 16, 32 };
	static const char *heads[4][3] = { { "cls_8", "cls_16", "cls_32" }, { "obj_8", "obj_16", "obj_32" },
		{ "bbox_8", "bbox_16", "bbox_32" }, { "kps_8", "kps_16", "kps_32" } };
	int cap = 4096, nf = 0, kept = 0;
	float *faces = malloc((size_t)cap * YUNET_FACE * sizeof(float));
	if (!faces) return 0;
	for (int i = 0; i < 3; i++) {
		const int cols = y->pw / strides[i], rows = y->ph / strides[i];
		const float *cls = yunet_tensor(y, heads[0][i], NULL), *obj = yunet_tensor(y, heads[1][i], NULL);
		const float *bbox = yunet_tensor(y, heads[2][i], NULL), *kps = yunet_tensor(y, heads[3][i], NULL);
		if (!cls || !obj || !bbox || !kps) { free(faces); return 0; }
		for (int r = 0; r < rows; r++)
			for (int c = 0; c < cols; c++) {
				const size_t idx = (size_t)r * cols + c;
				float cs = cls[idx], os = obj[idx];
				cs = cs < 0.f ? 0.f : cs > 1.f ? 1.f : cs;
				os = os < 0.f ? 0.f : os > 1.f ? 1.f : os;
				const float sc = sqrtf(cs * os);
				if (sc < score_thr) continue;
				if (nf == cap) { float *nfp = realloc(faces, (size_t)cap * 2 * YUNET_FACE * sizeof(float)); if (!nfp) break; faces = nfp; cap *= 2; }
				float *f = faces + (size_t)nf * YUNET_FACE;
				const float cx = (c + bbox[idx * 4 + 0]) * strides[i], cy = (r + bbox[idx * 4 + 1]) * strides[i];
				const float w = expf(bbox[idx * 4 + 2]) * strides[i], h = expf(bbox[idx * 4 + 3]) * strides[i];
				f[0] = cx - w / 2.f; f[1] = cy - h / 2.f; f[2] = w; f[3] = h;
				for (int n = 0; n < 5; n++) {
					f[4 + 2 * n] = (kps[idx * 10 + 2 * n] + c) * strides[i];
					f[4 + 2 * n + 1] = (kps[idx * 10 + 2 * n + 1] + r) * strides[i];
				}
				f[14] = sc;
				nf++;
			}
	}
	if (nf <= 1) {   /* OpenCV returns a lone face without NMS */
		if (nf == 1 && maxf > 0) { memcpy(out, faces, YUNET_FACE * sizeof(float)); kept = 1; }
	} else {
		/* NMSFast_: scores > threshold, stable sort descending, top_k, greedy keep while overlap <= nms */
		scored *sv = malloc((size_t)nf * sizeof *sv);
		int (*box)[4] = malloc((size_t)nf * sizeof *box);
		int *keep = malloc((size_t)nf * sizeof *keep);
		int ns = 0, nk = 0;
		if (sv && box && keep) {
			for (int i = 0; i < nf; i++) {
				const float *f = faces + (size_t)i * YUNET_FACE;
				box[i][0] = (int)f[0]; box[i][1] = (int)f[1]; box[i][2] = (int)f[2]; box[i][3] = (int)f[3];
				if (f[14] > score_thr) { sv[ns].score = f[14]; sv[ns].idx = i; ns++; }
			}
			qsort(sv, (size_t)ns, sizeof *sv, cmp_desc);
			if (top_k > 0 && top_k < ns) ns = top_k;
			for (int i = 0; i < ns; i++) {
				int ok = 1;
				for (int k = 0; k < nk && ok; k++) ok = overlap(box[sv[i].idx], box[keep[k]]) <= nms_thr;
				if (ok) keep[nk++] = sv[i].idx;
			}
			for (int k = 0; k < nk && kept < maxf; k++)
				memcpy(out + (size_t)kept++ * YUNET_FACE, faces + (size_t)keep[k] * YUNET_FACE, YUNET_FACE * sizeof(float));
		}
		free(sv); free(box); free(keep);
	}
	free(faces);
	y->optime[T_DECODE] = now() - t0;
	return kept;
}

const float *yunet_tensor(yunet *y, const char *name, size_t *n)
{
	int i = tindex(y, name);
	if (i < 0 || y->t[i].padded) return NULL;   /* only flat tensors (the heads) are readable as arrays */
	if (n) *n = y->t[i].n;
	return y->t[i].data;
}

int yunet_timing(yunet *y, char names[][24], double *secs, int max)
{
	static const char *extra[] = { "input", "decode" };
	int k = 0;
	for (int o = 0; o < T_N && k < max; o++)
		if (y->optime[o] > 0) { snprintf(names[k], 24, "%s", o < OP_NONE ? opname[o] : extra[o - OP_NONE]); secs[k++] = y->optime[o]; }
	return k;
}

void yunet_close(yunet *y)
{
	if (!y) return;
	for (int i = 0; i < y->nt; i++) if (y->t[i].own) free(y->t[i].data);
	free(y->t);
	free(y->st);
	onnx_free(&y->m);
	free(y);
}
