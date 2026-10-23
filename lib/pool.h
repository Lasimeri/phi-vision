/* pool.h: a persistent thread pool. pool_run hands out tasks 0..n-1 through an atomic counter; the caller works too and
 * returns when every task is done and every worker has left the run. Workers spin briefly between runs, then sleep. */
#ifndef POOL_H
#define POOL_H

typedef void (*pool_fn)(void *arg, int task);
typedef struct pool pool;

/* threads includes the caller; cpus (may be NULL) pins worker i to cpus[i] (the caller is not moved) */
pool *pool_open(int threads, const int *cpus);
void pool_run(pool *p, pool_fn fn, void *arg, int ntasks);
int pool_threads(const pool *p);
void pool_close(pool *p);

#endif
