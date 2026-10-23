/* phiv.h: the host side of phi-vision: post frames to the card's pvd through the shared host-memory region and take
 * its answers. Never blocks: phiv_post refuses while a frame is in flight, phiv_take returns 0 until an answer is in. */
#ifndef PHIV_H
#define PHIV_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct phiv phiv;

/* path: /dev/shm/phi-hostmem (NULL for that); part_mib: the part's offset (0 for the default); NULL on failure */
phiv *phiv_open(const char *path, unsigned part_mib, int w, int h, int stride, float score, float nms, int topk);

/* 1 while the card has loaded this session and its heartbeat moved within the last second */
int phiv_ready(phiv *p);

/* copies the picture into the next slot and posts it; its frame number, or 0 when the card is not ready or busy */
uint64_t phiv_post(phiv *p, const uint8_t *bgr, int stride);

/* the newest answer not yet taken: its frame number, faces written to out (at most maxf rows of 15 floats) and
 * counted in *nfaces, the card's detection time in *us; 0 when there is none */
uint64_t phiv_take(phiv *p, float *out, int maxf, int *nfaces, unsigned *us);

void phiv_close(phiv *p);

#ifdef __cplusplus
}
#endif

#endif
