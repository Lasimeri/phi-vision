/* phiv-proto.h: the layout of phi-vision's part of the shared host-memory region (docs/protocol.md). The host maps
 * it from /dev/shm/phi-hostmem, the card from /dev/phihost and through its DMA engine. Little-endian; offsets are
 * bytes from the part's start. Host-written and card-written fields sit on separate 64-byte lines. */
#ifndef PHIV_PROTO_H
#define PHIV_PROTO_H

#define PHIV_MAGIC 0x53495650u      /* "PVIS" */
#define PHIV_VERSION 1
#define PHIV_PART_MIB 1536          /* default part offset in the region: above the AVX-512 worker (0-768 MiB) and
                                       worldgen (1024 MiB + ~280 MiB) */

/* written by the host */
#define PHIV_H_MAGIC 0              /* u32 */
#define PHIV_H_VERSION 4            /* u32 */
#define PHIV_H_W 8                  /* u32 picture width */
#define PHIV_H_H 12                 /* u32 picture height */
#define PHIV_H_STRIDE 16            /* u32 bytes between rows (BGR) */
#define PHIV_H_SESSION 24           /* u64 raised by each host open */
#define PHIV_H_SCORE 32             /* f32 score threshold */
#define PHIV_H_NMS 36               /* f32 NMS threshold */
#define PHIV_H_TOPK 40              /* u32 */
#define PHIV_H_POSTED 64            /* u64 frames posted this session; frame n is in slot n % PHIV_SLOTS */

/* written by the card */
#define PHIV_C_SESSION 128          /* u64 the session the card has loaded */
#define PHIV_C_THREADS 136          /* u32 */
#define PHIV_C_STATE 140            /* u32 0 starting, 1 ready, 2 error */
#define PHIV_C_DONE 192             /* u64 frames answered this session */
#define PHIV_C_HEART 256            /* u64 counted up about every 100 ms */
#define PHIV_C_FRAMES 320           /* u64 frames answered since the daemon started */
#define PHIV_C_BUSY_US 328          /* u64 microseconds spent on frames (DMA + detection) since start */
#define PHIV_C_LAST_US 336          /* u32 the last frame's detection, microseconds */
#define PHIV_C_LAST_DMA_US 340      /* u32 the last frame's DMA pull, microseconds */

/* answers: entry n % PHIV_SLOTS */
#define PHIV_RESULTS (64u << 10)
#define PHIV_RESULT_SIZE 4096u
#define PHIV_R_SEQ 0                /* u64 n, written last */
#define PHIV_R_NFACES 8             /* u32 */
#define PHIV_R_US 12                /* u32 detection microseconds */
#define PHIV_R_FACES 64             /* nfaces x 15 f32: x, y, w, h, 5 landmarks, score (cv::FaceDetectorYN rows) */
#define PHIV_MAXF 64

/* frames */
#define PHIV_FRAMES (2u << 20)
#define PHIV_FRAME_SIZE (2u << 20)
#define PHIV_SLOTS 2
#define PHIV_PART_SIZE (PHIV_FRAMES + PHIV_SLOTS * PHIV_FRAME_SIZE)

#endif
