/*
 * pvd: the phi-vision daemon on the card. Takes frames from phi-vision's part of the shared host-memory region
 * (lib/phiv-proto.h, docs/protocol.md) through a card-owned DMA channel, runs YuNet on the vector unit, and writes the
 * faces back into the region.
 *
 * usage: pvd [-m MODEL] [-o PART_MIB] [-t THREADS] [-c DMA_CHANNEL]
 * Defaults: /tmp/face/yunet.onnx, 1536, 118 (two hardware threads on cores 0 to 58), channel 5.
 *
 * With a host-memory region under 4 GiB the card's init makes the whole region swap (/dev/phiblk1); pvd turns that
 * swap off first, as the AVX-512 worker does, since the region is then shared memory, not swap.
 */
#define _GNU_SOURCE
#include "cdma.h"
#include "../lib/phiv-proto.h"
#include "../lib/yunet.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/swap.h>
#include <time.h>
#include <unistd.h>

static volatile uint8_t *part;   /* the part's first PHIV_FRAMES bytes, uncached through /dev/phihost */

static void say(const char *fmt, ...)
{
	char ts[32];
	time_t t = time(NULL);
	strftime(ts, sizeof ts, "%F %T", localtime(&t));
	va_list ap;
	va_start(ap, fmt);
	fprintf(stderr, "%s pvd: ", ts);
	vfprintf(stderr, fmt, ap);
	fputc('\n', stderr);
	va_end(ap);
}

static double now(void)
{
	struct timespec s;
	clock_gettime(CLOCK_MONOTONIC, &s);
	return s.tv_sec + s.tv_nsec / 1e9;
}

static inline void fence(void) { __asm__ volatile("lock addq $0, (%%rsp)" ::: "memory"); }
static inline uint32_t rd32(int off) { return *(volatile uint32_t *)(part + off); }
static inline uint64_t rd64(int off) { return *(volatile uint64_t *)(part + off); }
static inline void wr32(int off, uint32_t v) { *(volatile uint32_t *)(part + off) = v; }
static inline void wr64(int off, uint64_t v) { *(volatile uint64_t *)(part + off) = v; }

static int swap_off(void)
{
	FILE *f = fopen("/proc/swaps", "r");
	char line[256];
	int on = 0;
	while (f && fgets(line, sizeof line, f)) on |= !strncmp(line, "/dev/phiblk1", 12);
	if (f) fclose(f);
	if (!on) return 0;
	if (swapoff("/dev/phiblk1")) { say("swapoff /dev/phiblk1: %s", strerror(errno)); return -1; }
	say("swap on /dev/phiblk1 turned off: the region is shared memory now");
	return 0;
}

static int hugepages(int need)
{
	FILE *f = fopen("/proc/meminfo", "r");
	char line[256];
	long total = 0, freep = 0;
	while (f && fgets(line, sizeof line, f)) {
		sscanf(line, "HugePages_Total: %ld", &total);
		sscanf(line, "HugePages_Free: %ld", &freep);
	}
	if (f) fclose(f);
	if (freep >= need) return 0;
	f = fopen("/proc/sys/vm/nr_hugepages", "w");
	if (!f) { say("nr_hugepages: %s", strerror(errno)); return -1; }
	fprintf(f, "%ld\n", total + (need - freep));
	fclose(f);
	return 0;
}

int main(int argc, char **argv)
{
	const char *model = "/tmp/face/yunet.onnx";
	uint64_t part_mib = PHIV_PART_MIB;
	int threads = 118, chan = 5, opt;
	while ((opt = getopt(argc, argv, "m:o:t:c:")) != -1) {
		if (opt == 'm') model = optarg;
		else if (opt == 'o') part_mib = strtoull(optarg, NULL, 10);
		else if (opt == 't') threads = atoi(optarg);
		else if (opt == 'c') chan = atoi(optarg);
		else { fprintf(stderr, "usage: pvd [-m MODEL] [-o PART_MIB] [-t THREADS] [-c DMA_CHANNEL]\n"); return 2; }
	}
	const uint64_t off = part_mib << 20;
	if (swap_off() || hugepages(2)) return 1;

	cdma D;
	if (cdma_open(&D, chan)) return 1;
	if (off + PHIV_PART_SIZE > D.region_size) {
		say("part at %llu MiB does not fit the %llu MiB region", (unsigned long long)part_mib, (unsigned long long)(D.region_size >> 20));
		return 1;
	}
	int fd = open("/dev/phihost", O_RDWR | O_SYNC);
	part = fd < 0 ? MAP_FAILED : mmap(NULL, PHIV_FRAMES, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)off);
	if (part == MAP_FAILED) { say("/dev/phihost: %s", strerror(errno)); return 1; }
	uint64_t land_phys;
	uint8_t *land = huge_alloc(PHIV_FRAME_SIZE, &land_phys, 1);
	if (!land) { say("no huge page for the landing area"); return 1; }

	int *cpus = calloc((size_t)threads, sizeof *cpus);
	for (int i = 0; i < threads - 1; i++) { int k = i + 1; cpus[i] = 1 + 4 * (k / 2) + (k % 2); }
	pool *pl = threads > 1 ? pool_open(threads, cpus) : NULL;

	wr32(PHIV_C_STATE, 0);
	wr32(PHIV_C_THREADS, (uint32_t)threads);
	wr64(PHIV_C_FRAMES, 0);
	wr64(PHIV_C_BUSY_US, 0);
	say("serving: part at %llu MiB (card address 0x%llx), DMA channel %d, %d threads, model %s", (unsigned long long)part_mib,
	    (unsigned long long)(D.region_addr + off), chan, threads, model);

	yunet *y = NULL;
	int w = 0, h = 0, stride = 0, topk = 50;
	float score = 0.6f, nms = 0.3f;
	uint64_t session = rd64(PHIV_C_SESSION), done = 0, heart = 0, frames = 0, busy_us = 0;
	double next_heart = 0;
	struct timespec nap = { 0, 50000 };
	float faces[PHIV_MAXF * YUNET_FACE];
	for (;;) {
		double t = now();
		if (t >= next_heart) { wr64(PHIV_C_HEART, ++heart); next_heart = t + 0.1; }
		/* a new session: the host's picture size and thresholds */
		uint64_t s = rd64(PHIV_H_SESSION);
		if (s != session && rd32(PHIV_H_MAGIC) == PHIV_MAGIC && rd32(PHIV_H_VERSION) == PHIV_VERSION) {
			int nw = (int)rd32(PHIV_H_W), nh = (int)rd32(PHIV_H_H), ns = (int)rd32(PHIV_H_STRIDE);
			uint32_t sb = rd32(PHIV_H_SCORE), nb = rd32(PHIV_H_NMS);
			memcpy(&score, &sb, 4);
			memcpy(&nms, &nb, 4);
			topk = (int)rd32(PHIV_H_TOPK);
			if (nw <= 0 || nh <= 0 || ns < nw * 3 || (uint64_t)ns * nh > PHIV_FRAME_SIZE) {
				say("session %llu: bad picture %dx%d stride %d", (unsigned long long)s, nw, nh, ns);
				wr32(PHIV_C_STATE, 2);
				session = s;
				continue;
			}
			if (nw != w || nh != h) {
				char err[256];
				yunet_close(y);
				y = yunet_open(model, nw, nh, pl, err, sizeof err);
				if (!y) { say("yunet: %s", err); wr32(PHIV_C_STATE, 2); session = s; continue; }
				w = nw; h = nh;
			}
			stride = ns;
			done = rd64(PHIV_H_POSTED);
			wr64(PHIV_C_DONE, done);
			fence();
			wr64(PHIV_C_SESSION, s);
			wr32(PHIV_C_STATE, 1);
			session = s;
			say("session %llu: %dx%d stride %d, score %.2f nms %.2f top %d", (unsigned long long)s, w, h, stride, score, nms, topk);
		}
		if (!y || s != session || rd64(PHIV_H_POSTED) <= done) { nanosleep(&nap, NULL); continue; }

		/* frame done + 1: pull it in pieces of at most 16383 lines, detect, answer */
		const uint64_t n = done + 1;
		const uint64_t src = D.region_addr + off + PHIV_FRAMES + (n % PHIV_SLOTS) * PHIV_FRAME_SIZE;
		const uint32_t bytes = (uint32_t)(((uint64_t)stride * h + 63) / 64 * 64);
		double t0 = now();
		uint64_t seq = 0;
		for (uint32_t o = 0; o < bytes; o += 512u << 10) {
			uint32_t len = bytes - o < (512u << 10) ? bytes - o : 512u << 10;
			seq = cdma_submit(&D, src + o, land_phys + o, len, 0, 0);
			if (!seq) { say("DMA submit refused"); return 1; }
		}
		if (cdma_wait(&D, seq, 2.0)) { say("DMA pull of frame %llu timed out", (unsigned long long)n); return 1; }
		double t1 = now();
		int nf = yunet_detect(y, land, stride, score, nms, topk, faces, PHIV_MAXF);
		double t2 = now();
		const int r = (int)(PHIV_RESULTS + (n % PHIV_SLOTS) * PHIV_RESULT_SIZE);
		for (int i = 0; i < nf * YUNET_FACE; i++) {
			uint32_t v;
			memcpy(&v, &faces[i], 4);
			wr32(r + PHIV_R_FACES + 4 * i, v);
		}
		wr32(r + PHIV_R_NFACES, (uint32_t)nf);
		wr32(r + PHIV_R_US, (uint32_t)((t2 - t1) * 1e6));
		fence();
		wr64(r + PHIV_R_SEQ, n);
		fence();
		wr64(PHIV_C_DONE, n);
		done = n;
		frames++;
		busy_us += (uint64_t)((t2 - t0) * 1e6);
		wr64(PHIV_C_FRAMES, frames);
		wr64(PHIV_C_BUSY_US, busy_us);
		wr32(PHIV_C_LAST_US, (uint32_t)((t2 - t1) * 1e6));
		wr32(PHIV_C_LAST_DMA_US, (uint32_t)((t1 - t0) * 1e6));
	}
}
