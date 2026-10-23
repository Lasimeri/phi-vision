/*
 * yunet-run: lib/yunet.c on raw BGR frames: the faces, per-op time, and with REFDIR the comparison with ocv-ref's
 * heads and faces.
 *
 * usage: yunet-run MODEL W H THREADS ROUNDS FRAME.bgr [REFDIR]
 * THREADS includes this one; on the card the workers are pinned to two hardware threads a core from core 0
 * ($YUNET_CPUS=list overrides).
 */
#include "../lib/yunet.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void)
{
	struct timespec s;
	clock_gettime(CLOCK_MONOTONIC, &s);
	return s.tv_sec + s.tv_nsec / 1e9;
}

static float *readf(const char *path, size_t *n)
{
	FILE *f = fopen(path, "rb");
	if (!f) return NULL;
	fseek(f, 0, SEEK_END);
	long len = ftell(f);
	fseek(f, 0, SEEK_SET);
	float *p = malloc((size_t)len + 4);
	*n = fread(p, 1, (size_t)len, f) / sizeof(float);
	fclose(f);
	return p;
}

int main(int argc, char **argv)
{
	if (argc < 7) { fprintf(stderr, "usage: yunet-run MODEL W H THREADS ROUNDS FRAME.bgr [REFDIR]\n"); return 2; }
	const int W = atoi(argv[2]), H = atoi(argv[3]), threads = atoi(argv[4]), rounds = atoi(argv[5]);
	uint8_t *img = malloc((size_t)W * H * 3);
	FILE *f = fopen(argv[6], "rb");
	if (!f || fread(img, 1, (size_t)W * H * 3, f) != (size_t)W * H * 3) { fprintf(stderr, "cannot read %s\n", argv[6]); return 1; }
	fclose(f);

	/* pinning: $YUNET_CPUS, else on a card (more than 200 CPUs) hardware threads 0 and 1 of cores 0, 1, ...
	 * (CPU 1 + 4c + t; CPU 0 is the OS core's first thread) */
	int *cpus = NULL;
	const char *env = getenv("YUNET_CPUS");
	long ncpu = 0;
	{ FILE *c = fopen("/proc/cpuinfo", "r"); char line[256]; if (c) { while (fgets(line, sizeof line, c)) ncpu += !strncmp(line, "processor", 9); fclose(c); } }
	if (threads > 1 && (env || ncpu > 200)) {
		cpus = calloc((size_t)threads, sizeof *cpus);
		if (env) { char *p = (char *)env; for (int i = 0; i < threads - 1 && *p; i++) { cpus[i] = (int)strtol(p, &p, 10); if (*p == ',') p++; } }
		else for (int i = 0; i < threads - 1; i++) { int k = i + 1; cpus[i] = 1 + 4 * (k / 2) + (k % 2); }
	}
	pool *pl = threads > 1 ? pool_open(threads, cpus) : NULL;
	char err[256];
	yunet *y = yunet_open(argv[1], W, H, pl, err, sizeof err);
	if (!y) { fprintf(stderr, "yunet: %s\n", err); return 1; }

	float faces[64 * YUNET_FACE];
	int nf = 0;
	double best = 1e9, sum = 0;
	nf = yunet_detect(y, img, W * 3, 0.6f, 0.3f, 50, faces, 64);   /* warm-up */
	for (int r = 0; r < rounds; r++) {
		double t0 = now();
		nf = yunet_detect(y, img, W * 3, 0.6f, 0.3f, 50, faces, 64);
		double t = now() - t0;
		sum += t;
		if (t < best) best = t;
	}
	printf("threads %d rounds %d: best %.3f ms mean %.3f ms, %d faces\n", threads, rounds, best * 1e3, rounds ? sum / rounds * 1e3 : 0, nf);
	char names[16][24]; double secs[16];
	int nt = yunet_timing(y, names, secs, 16);
	printf("last round by op:");
	for (int i = 0; i < nt; i++) printf(" %s %.3f", names[i], secs[i] * 1e3);
	printf(" ms\n");
	for (int i = 0; i < nf; i++) {
		const float *q = faces + i * YUNET_FACE;
		printf("face %d: %.2f %.2f %.2f %.2f score %.5f\n", i, q[0], q[1], q[2], q[3], q[14]);
	}

	int bad = 0;
	if (argc > 7) {
		static const char *heads[12] = { "cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32",
			"bbox_8", "bbox_16", "bbox_32", "kps_8", "kps_16", "kps_32" };
		for (int i = 0; i < 12; i++) {
			char path[4096];
			snprintf(path, sizeof path, "%s/%s.f32", argv[7], heads[i]);
			size_t nr = 0, nm = 0;
			float *ref = readf(path, &nr);
			const float *mine = yunet_tensor(y, heads[i], &nm);
			if (!ref || !mine || nr != nm) { printf("%-8s size mismatch (ref %zu, mine %zu)\n", heads[i], nr, nm); bad++; free(ref); continue; }
			double maxabs = 0, maxref = 0;
			for (size_t k = 0; k < nm; k++) {
				double d = fabs((double)ref[k] - mine[k]);
				if (d > maxabs) maxabs = d;
				if (fabs(ref[k]) > maxref) maxref = fabs(ref[k]);
			}
			printf("%-8s n %7zu  max |diff| %.3g  (max |ref| %.3g)\n", heads[i], nm, maxabs, maxref);
			free(ref);
		}
		char path[4096];
		snprintf(path, sizeof path, "%s/faces.txt", argv[7]);
		FILE *ff = fopen(path, "r");
		int nr = 0; float rf[64][YUNET_FACE];
		while (ff && nr < 64) { int k = 0; for (; k < YUNET_FACE; k++) if (fscanf(ff, "%f", &rf[nr][k]) != 1) break; if (k < YUNET_FACE) break; nr++; }
		if (ff) fclose(ff);
		double worst = 0, wscore = 0;
		for (int i = 0; i < nr && i < nf; i++)
			for (int k = 0; k < YUNET_FACE; k++) {
				double d = fabs((double)rf[i][k] - faces[i * YUNET_FACE + k]);
				if (k == 14) { if (d > wscore) wscore = d; } else if (d > worst) worst = d;
			}
		printf("faces: opencv %d, mine %d; worst box/landmark diff %.4f px, worst score diff %.3g\n", nr, nf, worst, wscore);
		if (nr != nf) bad++;
	}
	yunet_close(y);
	pool_close(pl);
	free(img);
	free(cpus);
	return bad ? 1 : 0;
}
