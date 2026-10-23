/*
 * onnxdump: print an ONNX model's graph (inputs, outputs, initializers, nodes with
 * attributes) by reading the protobuf wire format directly. No dependencies.
 *
 * usage: onnxdump MODEL.onnx
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const uint8_t *p, *end; } buf;

static uint64_t varint(buf *b)
{
	uint64_t v = 0;
	for (int s = 0; b->p < b->end; s += 7) {
		uint8_t c = *b->p++;
		v |= (uint64_t)(c & 0x7f) << s;
		if (!(c & 0x80))
			break;
	}
	return v;
}

/* next field: returns 0 at the end; sets *field, *wire, and for wire 2 the sub-buffer */
static int field(buf *b, int *f, int *wire, buf *sub, uint64_t *val)
{
	if (b->p >= b->end)
		return 0;
	uint64_t key = varint(b);
	*f = (int)(key >> 3);
	*wire = (int)(key & 7);
	switch (*wire) {
	case 0: *val = varint(b); break;
	case 1: memcpy(val, b->p, 8); b->p += 8; break;
	case 5: { uint32_t v; memcpy(&v, b->p, 4); *val = v; b->p += 4; break; }
	case 2: { uint64_t n = varint(b); sub->p = b->p; sub->end = b->p + n; b->p += n; break; }
	default: fprintf(stderr, "wire type %d\n", *wire); exit(1);
	}
	return 1;
}

static void pstr(buf s) { printf("%.*s", (int)(s.end - s.p), (const char *)s.p); }

static const char *dtype(int t)
{
	static const char *n[] = { "?", "f32", "u8", "i8", "u16", "i16", "i32", "i64", "str", "bool", "f16", "f64" };
	return t >= 0 && t < 12 ? n[t] : "?";
}

static void tensor(buf t)
{
	int f, w; buf s; uint64_t v;
	int64_t dims[8]; int nd = 0, type = 0; buf name = { 0, 0 }; size_t raw = 0, fl = 0, i64 = 0;
	while (field(&t, &f, &w, &s, &v)) {
		if (f == 1 && w == 0 && nd < 8) dims[nd++] = (int64_t)v;
		else if (f == 1 && w == 2) while (s.p < s.end && nd < 8) dims[nd++] = (int64_t)varint(&s);
		else if (f == 2) type = (int)v;
		else if (f == 8) name = s;
		else if (f == 9) raw = (size_t)(s.end - s.p);
		else if (f == 4) fl += w == 2 ? (size_t)(s.end - s.p) / 4 : 1;
		else if (f == 7) { if (w == 2) while (s.p < s.end) { varint(&s); i64++; } else i64++; }
	}
	printf("  init "); pstr(name); printf(" %s [", dtype(type));
	for (int i = 0; i < nd; i++) printf("%s%lld", i ? "," : "", (long long)dims[i]);
	printf("] raw %zu float_data %zu int64_data %zu\n", raw, fl, i64);
}

static void attribute(buf a)
{
	int f, w; buf s; uint64_t v; buf name = { 0, 0 };
	char out[512]; int o = 0; out[0] = 0;
	while (field(&a, &f, &w, &s, &v)) {
		if (f == 1) name = s;
		else if (f == 2) { float x; uint32_t u = (uint32_t)v; memcpy(&x, &u, 4); o += snprintf(out + o, sizeof out - o, "%g", x); }
		else if (f == 3) o += snprintf(out + o, sizeof out - o, "%lld", (long long)v);
		else if (f == 4) o += snprintf(out + o, sizeof out - o, "\"%.*s\"", (int)(s.end - s.p), (const char *)s.p);
		else if (f == 8) {
			o += snprintf(out + o, sizeof out - o, "[");
			if (w == 2) { int first = 1; while (s.p < s.end) { o += snprintf(out + o, sizeof out - o, "%s%lld", first ? "" : ",", (long long)varint(&s)); first = 0; } }
			else o += snprintf(out + o, sizeof out - o, "%lld", (long long)v);
			o += snprintf(out + o, sizeof out - o, "]");
		} else if (f == 7) {
			o += snprintf(out + o, sizeof out - o, "[");
			if (w == 2) { int first = 1; while (s.p + 4 <= s.end) { float x; memcpy(&x, s.p, 4); s.p += 4; o += snprintf(out + o, sizeof out - o, "%s%g", first ? "" : ",", x); first = 0; } }
			o += snprintf(out + o, sizeof out - o, "]");
		} else if (f == 5) o += snprintf(out + o, sizeof out - o, "<tensor>");
	}
	printf(" "); pstr(name); printf("=%s", out);
}

static void node(buf n)
{
	int f, w; buf s; uint64_t v;
	buf op = { 0, 0 }, name = { 0, 0 }, in[16], outs[8], at[16]; int ni = 0, no = 0, na = 0;
	while (field(&n, &f, &w, &s, &v)) {
		if (f == 1 && ni < 16) in[ni++] = s;
		else if (f == 2 && no < 8) outs[no++] = s;
		else if (f == 3) name = s;
		else if (f == 4) op = s;
		else if (f == 5 && na < 16) at[na++] = s;
	}
	printf("  node "); pstr(op); printf(" ("); pstr(name); printf(") in");
	for (int i = 0; i < ni; i++) { printf(" "); pstr(in[i]); }
	printf(" -> out");
	for (int i = 0; i < no; i++) { printf(" "); pstr(outs[i]); }
	printf(" |");
	for (int i = 0; i < na; i++) attribute(at[i]);
	printf("\n");
}

static void valueinfo(const char *kind, buf vi)
{
	int f, w; buf s; uint64_t v; buf name = { 0, 0 }; char shape[256]; int o = 0; shape[0] = 0;
	while (field(&vi, &f, &w, &s, &v)) {
		if (f == 1) name = s;
		else if (f == 2) {   /* TypeProto -> tensor_type (1) -> shape (2) -> dim (1) */
			int f2, w2; buf s2; uint64_t v2;
			while (field(&s, &f2, &w2, &s2, &v2)) if (f2 == 1) {
				int f3, w3; buf s3; uint64_t v3;
				while (field(&s2, &f3, &w3, &s3, &v3)) if (f3 == 2) {
					int f4, w4; buf s4; uint64_t v4;
					while (field(&s3, &f4, &w4, &s4, &v4)) if (f4 == 1) {
						int f5, w5; buf s5; uint64_t v5;
						while (field(&s4, &f5, &w5, &s5, &v5)) {
							if (f5 == 1) o += snprintf(shape + o, sizeof shape - o, "%lld,", (long long)v5);
							else if (f5 == 2) o += snprintf(shape + o, sizeof shape - o, "%.*s,", (int)(s5.end - s5.p), (const char *)s5.p);
						}
					}
				}
			}
		}
	}
	printf("  %s ", kind); pstr(name); printf(" [%s]\n", shape);
}

int main(int argc, char **argv)
{
	if (argc != 2) { fprintf(stderr, "usage: onnxdump MODEL.onnx\n"); return 2; }
	FILE *fp = fopen(argv[1], "rb");
	if (!fp) { perror(argv[1]); return 1; }
	fseek(fp, 0, SEEK_END); long n = ftell(fp); fseek(fp, 0, SEEK_SET);
	uint8_t *data = malloc(n);
	if (fread(data, 1, n, fp) != (size_t)n) { perror("read"); return 1; }
	fclose(fp);
	buf m = { data, data + n }, s, g; int f, w; uint64_t v; int have = 0;
	while (field(&m, &f, &w, &s, &v)) {
		if (f == 1) printf("ir_version %llu\n", (unsigned long long)v);
		else if (f == 8) { int f2, w2; buf s2; uint64_t v2; while (field(&s, &f2, &w2, &s2, &v2)) if (f2 == 2) printf("opset %llu\n", (unsigned long long)v2); }
		else if (f == 7) { g = s; have = 1; }
	}
	if (!have) { fprintf(stderr, "no graph\n"); return 1; }
	int nn = 0, ninit = 0;
	buf g2 = g;
	while (field(&g2, &f, &w, &s, &v)) {
		if (f == 11) valueinfo("input", s);
		else if (f == 12) valueinfo("output", s);
	}
	g2 = g;
	while (field(&g2, &f, &w, &s, &v)) if (f == 5) { tensor(s); ninit++; }
	g2 = g;
	while (field(&g2, &f, &w, &s, &v)) if (f == 1) { node(s); nn++; }
	printf("nodes %d initializers %d\n", nn, ninit);
	return 0;
}
