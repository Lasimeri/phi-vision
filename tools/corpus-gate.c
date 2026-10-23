/*
 * corpus-gate: lib/yunet.c against OpenCV on a corpus written by facetrack ($FACETRACK_CORPUS): every
 * fNNNNNN_WxH.bgr (the detector's input) beside fNNNNNN.det (OpenCV FaceDetectorYN's rows for it).
 *
 * usage: corpus-gate MODEL DIR THREADS [PX_TOL SCORE_TOL]
 * A frame passes when it has the same number of faces as OpenCV and every row's box and landmarks are within
 * PX_TOL pixels (default 0.5) and its score within SCORE_TOL (default 1e-3). Faces whose score is within SCORE_TOL
 * of the 0.6 threshold may appear on one side only; those frames are reported as borderline, not failures.
 */
#include "../lib/yunet.h"

#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXF 64

static int cmpname(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

static double now(void)
{
	struct timespec s;
	clock_gettime(CLOCK_MONOTONIC, &s);
	return s.tv_sec + s.tv_nsec / 1e9;
}

int main(int argc, char **argv)
{
	if (argc < 4) { fprintf(stderr, "usage: corpus-gate MODEL DIR THREADS [PX_TOL SCORE_TOL]\n"); return 2; }
	const int threads = atoi(argv[3]);
	const double ptol = argc > 4 ? atof(argv[4]) : 0.5, stol = argc > 5 ? atof(argv[5]) : 1e-3;
	DIR *d = opendir(argv[2]);
	if (!d) { perror(argv[2]); return 1; }
	char **names = NULL;
	int n = 0, cap = 0;
	struct dirent *e;
	while ((e = readdir(d))) {
		size_t l = strlen(e->d_name);
		if (l > 4 && !strcmp(e->d_name + l - 4, ".bgr")) {
			if (n == cap) { cap = cap ? cap * 2 : 256; names = realloc(names, (size_t)cap * sizeof *names); }
			names[n++] = strdup(e->d_name);
		}
	}
	closedir(d);
	qsort(names, (size_t)n, sizeof *names, cmpname);

	int *cpus = NULL;
	long ncpu = 0;
	{ FILE *c = fopen("/proc/cpuinfo", "r"); char line[256]; if (c) { while (fgets(line, sizeof line, c)) ncpu += !strncmp(line, "processor", 9); fclose(c); } }
	if (threads > 1 && ncpu > 200) {
		cpus = calloc((size_t)threads, sizeof *cpus);
		for (int i = 0; i < threads - 1; i++) { int k = i + 1; cpus[i] = 1 + 4 * (k / 2) + (k % 2); }
	}
	pool *pl = threads > 1 ? pool_open(threads, cpus) : NULL;

	yunet *y = NULL;
	int cw = 0, ch = 0, pass = 0, fail = 0, border = 0, withfaces = 0, totfaces = 0;
	double worst_px = 0, worst_sc = 0, tsum = 0;
	uint8_t *img = NULL;
	for (int i = 0; i < n; i++) {
		long num; int w, h;
		if (sscanf(names[i], "f%ld_%dx%d.bgr", &num, &w, &h) != 3) continue;
		if (w != cw || h != ch) {
			char err[256];
			yunet_close(y);
			y = yunet_open(argv[1], w, h, pl, err, sizeof err);
			if (!y) { fprintf(stderr, "yunet: %s\n", err); return 1; }
			cw = w; ch = h;
			free(img);
			img = malloc((size_t)w * h * 3);
		}
		char path[4096];
		snprintf(path, sizeof path, "%s/%s", argv[2], names[i]);
		FILE *f = fopen(path, "rb");
		if (!f || fread(img, 1, (size_t)w * h * 3, f) != (size_t)w * h * 3) { if (f) fclose(f); continue; }
		fclose(f);
		float ref[MAXF][YUNET_FACE];
		int nr = 0;
		snprintf(path, sizeof path, "%s/f%06ld.det", argv[2], num);
		f = fopen(path, "r");
		while (f && nr < MAXF) {
			int k = 0;
			for (; k < YUNET_FACE; k++) if (fscanf(f, "%f", &ref[nr][k]) != 1) break;
			if (k < YUNET_FACE) break;
			nr++;
		}
		if (f) fclose(f);

		float mine[MAXF * YUNET_FACE];
		double t0 = now();
		int nm = yunet_detect(y, img, w * 3, 0.6f, 0.3f, 50, mine, MAXF);
		tsum += now() - t0;
		withfaces += nr > 0;
		totfaces += nr;

		double px = 0, sc = 0;
		int ok = nm == nr;
		for (int a = 0; a < nr && a < nm; a++)
			for (int k = 0; k < YUNET_FACE; k++) {
				double dd = fabs((double)ref[a][k] - mine[a * YUNET_FACE + k]);
				if (k == 14) { if (dd > sc) sc = dd; } else if (dd > px) px = dd;
			}
		if (px > ptol || sc > stol) ok = 0;
		if (ok) { pass++; if (px > worst_px) worst_px = px; if (sc > worst_sc) worst_sc = sc; continue; }
		/* a count difference explained by a face at the threshold is borderline */
		int borderline = 0;
		if (nm != nr) {
			for (int a = 0; a < nr; a++) if (fabs(ref[a][14] - 0.6) <= stol) borderline = 1;
			for (int a = 0; a < nm; a++) if (fabs(mine[a * YUNET_FACE + 14] - 0.6) <= stol) borderline = 1;
		}
		if (borderline) border++; else fail++;
		printf("%s %s: opencv %d faces, mine %d; worst %.4f px, score %.3g\n", borderline ? "BORDER" : "FAIL", names[i], nr, nm, px, sc);
		for (int a = 0; a < nr; a++) printf("   opencv %d: %.2f %.2f %.2f %.2f score %.5f\n", a, ref[a][0], ref[a][1], ref[a][2], ref[a][3], ref[a][14]);
		for (int a = 0; a < nm; a++) { const float *q = mine + a * YUNET_FACE; printf("   mine   %d: %.2f %.2f %.2f %.2f score %.5f\n", a, q[0], q[1], q[2], q[3], q[14]); }
	}
	printf("frames %d (with faces %d, faces %d): pass %d, borderline %d, fail %d; worst passing %.4f px, score %.3g; "
	       "%.3f ms a frame (%d threads)\n", pass + border + fail, withfaces, totfaces, pass, border, fail, worst_px, worst_sc,
	       pass + border + fail ? tsum / (pass + border + fail) * 1e3 : 0, threads);
	yunet_close(y);
	pool_close(pl);
	return fail ? 1 : 0;
}
