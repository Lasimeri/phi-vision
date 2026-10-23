/*
 * pvtest: drive the card's pvd through the shared region with a corpus (fNNNNNN_WxH.bgr + fNNNNNN.det, see
 * corpus-gate) and check every answer against OpenCV's rows: the same faces, boxes and landmarks within 0.5 px.
 * Prints the round trip (post to answer) and the frames a second.
 *
 * usage: pvtest DIR [ROUNDS]
 */
#include "../host/phiv.h"

#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int cmpname(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

static double now(void)
{
	struct timespec s;
	clock_gettime(CLOCK_MONOTONIC, &s);
	return s.tv_sec + s.tv_nsec / 1e9;
}

int main(int argc, char **argv)
{
	if (argc < 2) { fprintf(stderr, "usage: pvtest DIR [ROUNDS]\n"); return 2; }
	const int rounds = argc > 2 ? atoi(argv[2]) : 1;
	DIR *d = opendir(argv[1]);
	if (!d) { perror(argv[1]); return 1; }
	char **names = NULL;
	int n = 0, cap = 0, w = 0, h = 0;
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
	if (!n || sscanf(names[0], "f%*d_%dx%d.bgr", &w, &h) != 2) { fprintf(stderr, "no frames\n"); return 1; }

	uint8_t *frames = malloc((size_t)n * w * h * 3);
	float (*ref)[64][15] = calloc((size_t)n, sizeof *ref);
	int *nref = calloc((size_t)n, sizeof *nref);
	for (int i = 0; i < n; i++) {
		long num;
		char path[4096];
		sscanf(names[i], "f%ld_", &num);
		snprintf(path, sizeof path, "%s/%s", argv[1], names[i]);
		FILE *f = fopen(path, "rb");
		if (!f || fread(frames + (size_t)i * w * h * 3, 1, (size_t)w * h * 3, f) != (size_t)w * h * 3) { fprintf(stderr, "read %s\n", path); return 1; }
		fclose(f);
		snprintf(path, sizeof path, "%s/f%06ld.det", argv[1], num);
		f = fopen(path, "r");
		while (f && nref[i] < 64) {
			int k = 0;
			for (; k < 15; k++) if (fscanf(f, "%f", &ref[i][nref[i]][k]) != 1) break;
			if (k < 15) break;
			nref[i]++;
		}
		if (f) fclose(f);
	}

	phiv *p = phiv_open(NULL, 0, w, h, w * 3, 0.6f, 0.3f, 50);
	if (!p) { fprintf(stderr, "phiv_open failed\n"); return 1; }
	double t0 = now();
	while (!phiv_ready(p)) if (now() - t0 > 3) { fprintf(stderr, "pvd did not take the session within 3 s\n"); return 1; }
	printf("session up in %.1f ms, %d frames %dx%d\n", (now() - t0) * 1e3, n, w, h);

	int fail = 0, total = 0;
	double worst = 0, rt_sum = 0, rt_max = 0, us_sum = 0;
	double start = now();
	for (int r = 0; r < rounds; r++)
		for (int i = 0; i < n; i++) {
			double tp = now();
			uint64_t seq;
			while (!(seq = phiv_post(p, frames + (size_t)i * w * h * 3, w * 3)))
				if (now() - tp > 2) { fprintf(stderr, "post refused for 2 s at frame %d\n", i); return 1; }
			float out[64 * 15];
			int nf = 0;
			unsigned us = 0;
			uint64_t got;
			while ((got = phiv_take(p, out, 64, &nf, &us)) != seq)
				if (now() - tp > 2) { fprintf(stderr, "no answer for frame %d within 2 s\n", i); return 1; }
			double rt = now() - tp;
			rt_sum += rt;
			us_sum += us;
			if (rt > rt_max) rt_max = rt;
			total++;
			double px = 0;
			int ok = nf == nref[i];
			for (int a = 0; a < nf && a < nref[i]; a++)
				for (int k = 0; k < 14; k++) { double dd = fabs(out[a * 15 + k] - ref[i][a][k]); if (dd > px) px = dd; }
			if (px > 0.5) ok = 0;
			if (!ok) { fail++; printf("FAIL %s: opencv %d faces, card %d, worst %.4f px\n", names[i], nref[i], nf, px); }
			else if (px > worst) worst = px;
		}
	double el = now() - start;
	printf("answers %d: fail %d, worst %.4f px; round trip mean %.3f ms max %.3f ms; card detection mean %.3f ms; %.1f frames/s\n",
	       total, fail, worst, rt_sum / total * 1e3, rt_max * 1e3, us_sum / total / 1e3, total / el);
	phiv_close(p);
	return fail ? 1 : 0;
}
