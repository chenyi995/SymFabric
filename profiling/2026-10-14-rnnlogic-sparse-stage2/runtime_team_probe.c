#define _GNU_SOURCE
#include <sched.h>
#include <stdint.h>
#include <stdatomic.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

typedef int (*get_int_fn)(void);
typedef void (*parallel_fn)(void (*)(void *), void *, unsigned, unsigned);
typedef struct {
    int index, tid, team_size, cpu_start, cpu_end, affinity_count;
    int affinity_cpus[CPU_SETSIZE];
    int64_t cpu_ns, wall_start_ns, wall_end_ns;
    uint64_t checksum;
} record_t;
typedef struct {
    get_int_fn thread_num;
    get_int_fn num_threads;
    record_t *records;
    atomic_int arrived;
    int capacity;
} context_t;

static int64_t ns_now(void) {
    struct timespec ts; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}
static int64_t wall_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}
static void worker(void *ptr) {
    context_t *ctx = (context_t *)ptr;
    int i = ctx->thread_num(), team = ctx->num_threads();
    if (i >= ctx->capacity) return;
    record_t *r = &ctx->records[i];
    r->index = i; r->tid = (int)syscall(SYS_gettid); r->team_size = team;
    r->cpu_start = sched_getcpu();
    cpu_set_t cpus; CPU_ZERO(&cpus); sched_getaffinity(0, sizeof(cpus), &cpus);
    r->affinity_count = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
        if (CPU_ISSET(cpu, &cpus)) r->affinity_cpus[r->affinity_count++] = cpu;
    atomic_fetch_add_explicit(&ctx->arrived, 1, memory_order_release);
    while (atomic_load_explicit(&ctx->arrived, memory_order_acquire) < team) {}
    r->wall_start_ns = wall_ns();
    int64_t start = ns_now();
    volatile uint64_t value = (uint64_t)i + 1;
    for (int k = 0; k < 1500000; ++k) value = value * 6364136223846793005ULL + 1;
    r->cpu_ns = ns_now() - start; r->wall_end_ns = wall_ns(); r->checksum = value;
    r->cpu_end = sched_getcpu();
}
int probe_runtime(void *parallel, void *thread_num, void *num_threads,
                  unsigned requested, record_t *records, int capacity) {
    context_t ctx = {(get_int_fn)thread_num, (get_int_fn)num_threads, records, 0, capacity};
    ((parallel_fn)parallel)(worker, &ctx, requested, 0);
    return atomic_load(&ctx.arrived);
}
