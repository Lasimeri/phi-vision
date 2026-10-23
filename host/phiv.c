/* phiv.c: see phiv.h and docs/protocol.md. */
#define _GNU_SOURCE
#include "phiv.h"
#include "../lib/phiv-proto.h"

#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

struct phiv {
	uint8_t *part;
	int w, h, stride;
	uint64_t session, posted, taken;
	uint64_t heart;
	double heart_seen;
};

static double now(void)
{
	struct timespec s;
	clock_gettime(CLOCK_MONOTONIC, &s);
	return s.tv_sec + s.tv_nsec / 1e9;
}

#define LD32(p, o) __atomic_load_n((uint32_t *)((p)->part + (o)), __ATOMIC_ACQUIRE)
#define LD64(p, o) __atomic_load_n((uint64_t *)((p)->part + (o)), __ATOMIC_ACQUIRE)
#define ST32(p, o, v) __atomic_store_n((uint32_t *)((p)->part + (o)), (v), __ATOMIC_RELEASE)
#define ST64(p, o, v) __atomic_store_n((uint64_t *)((p)->part + (o)), (v), __ATOMIC_RELEASE)

phiv *phiv_open(const char *path, unsigned part_mib, int w, int h, int stride, float score, float nms, int topk)
{
	if ((uint64_t)stride * h > PHIV_FRAME_SIZE) return NULL;
	int fd = open(path ? path : "/dev/shm/phi-hostmem", O_RDWR);
	if (fd < 0) return NULL;
	off_t len = lseek(fd, 0, SEEK_END);
	off_t off = (off_t)(part_mib ? part_mib : PHIV_PART_MIB) << 20;
	if (len < off + (off_t)PHIV_PART_SIZE) { close(fd); return NULL; }
	uint8_t *m = mmap(NULL, PHIV_PART_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, off);
	close(fd);
	if (m == MAP_FAILED) return NULL;
	phiv *p = calloc(1, sizeof *p);
	if (!p) { munmap(m, PHIV_PART_SIZE); return NULL; }
	p->part = m;
	p->w = w; p->h = h; p->stride = stride;
	ST32(p, PHIV_H_MAGIC, PHIV_MAGIC);
	ST32(p, PHIV_H_VERSION, PHIV_VERSION);
	ST32(p, PHIV_H_W, (uint32_t)w);
	ST32(p, PHIV_H_H, (uint32_t)h);
	ST32(p, PHIV_H_STRIDE, (uint32_t)stride);
	uint32_t v;
	memcpy(&v, &score, 4); ST32(p, PHIV_H_SCORE, v);
	memcpy(&v, &nms, 4); ST32(p, PHIV_H_NMS, v);
	ST32(p, PHIV_H_TOPK, (uint32_t)topk);
	ST64(p, PHIV_H_POSTED, 0);
	/* a session number no earlier open used: the time in microseconds */
	struct timespec t;
	clock_gettime(CLOCK_REALTIME, &t);
	p->session = (uint64_t)t.tv_sec * 1000000u + (uint64_t)t.tv_nsec / 1000u;
	ST64(p, PHIV_H_SESSION, p->session);
	p->heart = LD64(p, PHIV_C_HEART);
	p->heart_seen = now();
	return p;
}

int phiv_ready(phiv *p)
{
	uint64_t h = LD64(p, PHIV_C_HEART);
	double t = now();
	if (h != p->heart) { p->heart = h; p->heart_seen = t; }
	return t - p->heart_seen < 1.0 && LD64(p, PHIV_C_SESSION) == p->session && LD32(p, PHIV_C_STATE) == 1;
}

uint64_t phiv_post(phiv *p, const uint8_t *bgr, int stride)
{
	if (!phiv_ready(p) || LD64(p, PHIV_C_DONE) < p->posted) return 0;   /* one frame in flight at a time */
	const uint64_t n = p->posted + 1;
	uint8_t *dst = p->part + PHIV_FRAMES + (n % PHIV_SLOTS) * PHIV_FRAME_SIZE;
	for (int r = 0; r < p->h; r++) memcpy(dst + (size_t)r * p->stride, bgr + (size_t)r * stride, (size_t)p->w * 3);
	ST64(p, PHIV_H_POSTED, n);
	p->posted = n;
	return n;
}

uint64_t phiv_take(phiv *p, float *out, int maxf, int *nfaces, unsigned *us)
{
	const uint64_t d = LD64(p, PHIV_C_DONE);
	if (d <= p->taken || d > p->posted) return 0;
	const uint8_t *r = p->part + PHIV_RESULTS + (d % PHIV_SLOTS) * PHIV_RESULT_SIZE;
	if (__atomic_load_n((const uint64_t *)(r + PHIV_R_SEQ), __ATOMIC_ACQUIRE) != d) return 0;
	int n = (int)__atomic_load_n((const uint32_t *)(r + PHIV_R_NFACES), __ATOMIC_ACQUIRE);
	if (n > PHIV_MAXF) n = PHIV_MAXF;
	if (n > maxf) n = maxf;
	memcpy(out, r + PHIV_R_FACES, (size_t)n * 15 * sizeof(float));
	*nfaces = n;
	if (us) *us = __atomic_load_n((const uint32_t *)(r + PHIV_R_US), __ATOMIC_ACQUIRE);
	p->taken = d;
	return d;
}

void phiv_close(phiv *p)
{
	if (!p) return;
	munmap(p->part, PHIV_PART_SIZE);
	free(p);
}
