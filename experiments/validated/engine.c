/* Internal CLI for runner.py. JSON validation/provenance is performed there.
 * Only bounded child invocations are supported. No checkpoint/resume.
 */
#define _POSIX_C_SOURCE 200809L
#include "core.h"
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>

_Static_assert(sizeof(double) == 8 && DBL_MANT_DIG == 53 && CHAR_BIT == 8, "binary64 required");
static volatile sig_atomic_t stopped;
static void stop_signal(int signum) { stopped = signum; }
static double timer(clockid_t kind) {
    struct timespec t;
    if (clock_gettime(kind, &t)) return NAN;
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}
static int parse_u64(const char *s, uint64_t *out) {
    char *end; errno = 0;
    if (!*s || *s == '-' || *s == '+') return -1;
    unsigned long long n = strtoull(s, &end, 10);
    if (*end || errno) return -1;
    *out = (uint64_t)n; return 0;
}
static int parse_double(const char *s, double *out) {
    char *end; errno = 0;
    double n = strtod(s, &end);
    if (!*s || *end || errno || !isfinite(n)) return -1;
    *out = n; return 0;
}
typedef struct { uint64_t n; long double mean, m2; } Moment;
static void update(Moment *m, double x) {
    m->n++; long double d = x - m->mean;
    m->mean += d / m->n; m->m2 += d * (x - m->mean);
}
static int path_for(char *out, size_t size, const char *dir, const char *name) {
    int n = snprintf(out, size, "%s/%s", dir, name);
    return n < 0 || (size_t)n >= size ? -1 : 0;
}
static int durable_close(FILE *f) {
    int rc = fflush(f);
    if (!rc) rc = fsync(fileno(f));
    if (fclose(f)) rc = -1;
    return rc;
}
static int sync_directory(const char *dir) {
    int fd = open(dir, O_RDONLY);
    if (fd < 0) return -1;
    int rc = fsync(fd);
    if (close(fd)) rc = -1;
    return rc;
}
static uint32_t crc_table[256];
static void init_crc(void) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int bit = 0; bit < 8; bit++) c = (c >> 1u) ^ ((c & 1u) ? UINT32_C(0xedb88320) : 0);
        crc_table[i] = c;
    }
}
typedef struct {
    const char *dir; FILE *fp; uint64_t sequence, width, kind, threshold_count;
    uint64_t used, budget, chunk_size, rows, chunk_bytes, first, last;
    uint64_t published, chunks, fail_after, physical_written;
    uint32_t crc; int broken;
    char part[4096], final[4096], meta[4096], temp[4096];
} Writer;
static void u64le(unsigned char *p, uint64_t x) {
    for (int i = 0; i < 8; i++) p[i] = (unsigned char)(x >> (8*i));
}
static void u32le(unsigned char *p, uint32_t x) {
    for (int i = 0; i < 4; i++) p[i] = (unsigned char)(x >> (8*i));
}
static void f64le(unsigned char *p, double x) {
    uint64_t bits; memcpy(&bits, &x, 8); u64le(p, bits);
}
static int write_bytes(Writer *w, const unsigned char *p, size_t n) {
    if (w->fail_after && w->physical_written + n > w->fail_after) {
        size_t allowed = w->physical_written < w->fail_after ? (size_t)(w->fail_after - w->physical_written) : 0;
        if (allowed) (void)fwrite(p, 1, allowed, w->fp);
        w->physical_written += allowed; w->broken = 1; errno = ENOSPC; return -1;
    }
    if (fwrite(p, 1, n, w->fp) != n) { w->broken = 1; return -1; }
    for (size_t i = 0; i < n; i++) w->crc = crc_table[(w->crc ^ p[i]) & 255] ^ (w->crc >> 8u);
    w->physical_written += n; w->used += n; w->chunk_bytes += n;
    return 0;
}
static int finish_chunk(Writer *w) {
    if (!w->fp) return 0;
    FILE *fp = w->fp; w->fp = NULL;
    if (w->broken) { (void)fclose(fp); return -1; }
    if (durable_close(fp) || rename(w->part, w->final)) { w->broken = 1; return -1; }
    FILE *meta = fopen(w->temp, "wx");
    if (!meta) { w->broken = 1; return -1; }
    const char *base = strrchr(w->final, '/') + 1;
    int n = fprintf(meta, "{\"schema_version\":1,\"file\":\"%s\",\"records\":%" PRIu64
        ",\"bytes\":%" PRIu64 ",\"first_index\":%" PRIu64 ",\"last_index\":%" PRIu64
        ",\"record_bytes\":%" PRIu64 ",\"crc32\":\"%08" PRIx32 "\"}\n",
        base, w->rows, w->chunk_bytes, w->first, w->last, w->width, w->crc ^ UINT32_C(0xffffffff));
    if (n < 0) { (void)fclose(meta); w->broken = 1; return -1; }
    if (durable_close(meta) || rename(w->temp, w->meta) || sync_directory(w->dir)) {
        w->broken = 1; return -1;
    }
    w->used += (uint64_t)n; w->published += w->rows; w->chunks++; w->rows = 0;
    return 0;
}
static int start_chunk(Writer *w) {
    char name[100];
    snprintf(name, sizeof(name), "chunk-%08" PRIu64 ".bin", w->sequence++);
    if (path_for(w->final,sizeof(w->final),w->dir,name)) return -1;
    if (snprintf(w->part,sizeof(w->part),"%s.part",w->final) >= (int)sizeof(w->part) ||
        snprintf(w->meta,sizeof(w->meta),"%s.meta.json",w->final) >= (int)sizeof(w->meta) ||
        snprintf(w->temp,sizeof(w->temp),"%s.tmp",w->meta) >= (int)sizeof(w->temp)) return -1;
    w->fp = fopen(w->part, "wbx");
    if (!w->fp) return -1;
    w->crc = UINT32_C(0xffffffff); w->chunk_bytes = 0; w->rows = 0;
    unsigned char header[32] = {'T','A','X','I','v','1',0,0};
    u32le(header+8, (uint32_t)w->kind); u32le(header+12, (uint32_t)w->threshold_count);
    u64le(header+16,w->width); u64le(header+24,0);
    return write_bytes(w,header,sizeof(header));
}
/* 1 means storage ceiling, -1 means I/O failure. Metadata <=768 bytes/chunk. */
static int record(Writer *w, const unsigned char *data, uint64_t index) {
    uint64_t needed = w->width + 1024 + (w->fp ? 0 : 32);
    /* Reserve 512 bytes per finalized chunk for the Python SHA256 index. */
    uint64_t charged = w->used + 512*w->chunks;
    if (needed > w->budget || charged > w->budget - needed) return 1;
    if (!w->fp && start_chunk(w)) return -1;
    if (write_bytes(w,data,(size_t)w->width)) return -1;
    if (!w->rows) w->first = index;
    w->last = index; w->rows++;
    if (w->rows == w->chunk_size && finish_chunk(w)) return -1;
    return 0;
}
static int status_file(const char *dir, const char *state, uint64_t attempts, uint64_t retained,
                       double wall) {
    char path[4096], tmp[4096];
    if (path_for(path,sizeof(path),dir,"status.json") ||
        path_for(tmp,sizeof(tmp),dir,"status.json.tmp")) return -1;
    FILE *f = fopen(tmp,"w");
    if (!f) return -1;
    int n = fprintf(f,"{\"status\":\"%s\",\"iterations_completed\":%" PRIu64
                    ",\"finalized_records\":%" PRIu64 ",\"wall_seconds\":%.9g}\n",state,attempts,retained,wall);
    if (n < 0) { (void)fclose(f); return -1; }
    return durable_close(f) || rename(tmp,path) ? -1 : 0;
}
int main(int argc, char **argv) {
    if (argc != 23) { fputs("Internal engine: use python -m experiments.validated.runner\n",stderr); return 2; }
    int direct = !strcmp(argv[1],"direct"), suffix = !strcmp(argv[1],"suffix");
    int positional = !strcmp(argv[1],"positional2"), uniform = !strcmp(argv[1],"uniform2");
    int conditional = !strcmp(argv[1],"conditional") || !strcmp(argv[1],"conditional_sequential");
    int sequential = !strcmp(argv[1],"conditional_sequential");
    TaxiModel m; double beta, floor, scale, wall_limit, status_seconds;
    uint64_t iterations, warmup, stride, seed, stream, max_steps, chunk_records, budget, fail_after;
    if ((!direct && !suffix && !positional && !uniform && !conditional) ||
        parse_double(argv[2],&m.rate) || parse_double(argv[3],&m.L) || parse_double(argv[4],&m.c) ||
        parse_double(argv[5],&m.b) || parse_double(argv[6],&beta) || parse_u64(argv[7],&iterations) ||
        parse_u64(argv[8],&warmup) || parse_u64(argv[9],&stride) || parse_u64(argv[10],&seed) ||
        parse_u64(argv[11],&stream) || parse_u64(argv[12],&max_steps) ||
        parse_double(argv[13],&floor) || parse_double(argv[14],&scale) ||
        parse_double(argv[16],&wall_limit) || parse_u64(argv[17],&chunk_records) ||
        parse_u64(argv[18],&budget) || parse_double(argv[19],&status_seconds) ||
        parse_u64(argv[22],&fail_after) || !taxi_model_valid(m,beta) ||
        !iterations || !stride || !chunk_records || !(wall_limit>0) || !(status_seconds>0) ||
        warmup > UINT64_MAX-iterations || max_steps > SIZE_MAX/sizeof(double) ||
        ((positional || uniform) ? max_steps<2 : max_steps!=0) ||
        ((direct || conditional) && (beta!=0 || warmup!=0))) {
        fputs("invalid internal engine arguments\n",stderr); return 2;
    }
    double thresholds[64]; size_t nt = 0;
    const char *p = argv[21];
    while (*p && nt < 64) {
        char *end; errno = 0; double z = strtod(p,&end);
        if (p==end || errno || !isfinite(z) || z<0 || (*end && *end!=',') ||
            (nt && z<=thresholds[nt-1])) return 2;
        thresholds[nt++] = z; p = *end ? end+1 : end;
    }
    if (!nt || *p) return 2;
    struct sigaction action; memset(&action,0,sizeof(action));
    action.sa_handler = stop_signal; sigemptyset(&action.sa_mask);
    if (sigaction(SIGTERM,&action,NULL) || sigaction(SIGINT,&action,NULL)) return 2;
    TaxiRng rng;
    if (taxi_seed(&rng,seed,stream)) return 2;
    TaxiRng initial = rng;
    TaxiTrip cur = {0}, proposal = {0};
    double *cdf = NULL;
    Writer w = {0};
    w.dir=argv[20];w.width=conditional ? 40+8*nt : 48;w.kind=conditional?2:1;
    w.threshold_count=nt;w.budget=budget;w.chunk_size=chunk_records;w.fail_after=fail_after;
    init_crc();
    double start=timer(CLOCK_MONOTONIC), cpu_start=timer(CLOCK_PROCESS_CPUTIME_ID), last_status=start;
    Moment cost={0}, steps={0}, overshoot={0}, tails[64]={{0}};
    uint64_t attempts=0, accepted=0, changed=0, unfinished=0, atoms=0, tail_underflows[64]={0};
    const char *state="completed", *reason="requested_count"; int code=0;
    if (status_file(w.dir,"running",0,0,0)) { state="failed";reason="status_write_failure";code=74;goto done; }
    if (positional) {
        cdf=malloc((size_t)max_steps*sizeof(double));
        if (!cdf || taxi_position_cdf(cdf,(size_t)max_steps,floor,scale)) {
            state="failed";reason="position_allocation_or_numeric_error";code=70;goto done;
        }
    }
    if (!direct && !conditional && taxi_initialize(&cur,m,&rng,(size_t)max_steps,argv[15])) {
        state="failed";reason="initialization_or_allocation_error";code=70;goto done;
    }
    for (uint64_t t=0; t<warmup+iterations; t++) {
        if (stopped) { state="interrupted";reason=stopped==SIGINT?"SIGINT":"SIGTERM";code=128+stopped;break; }
        if ((t&255u)==0) {
            double now=timer(CLOCK_MONOTONIC);
            if (now-start>=wall_limit) { state="interrupted";reason="runtime_limit";code=75;break; }
            if (now-last_status>=status_seconds) {
                if (status_file(w.dir,"running",attempts,w.published,now-start)) {
                    state="failed";reason="status_write_failure";code=74;break;
                }
                last_status=now;
            }
        }
        int accept=1, cost_changed=1; TaxiPrefix prefix={0};
        if (conditional) {
            if (taxi_prefix(&proposal,m,&rng,sequential,&prefix)) {
                state="failed";reason="prefix_numerical_or_allocation_error";code=70;break;
            }
        } else if (direct) {
            if (taxi_generate(&cur,0,m,&rng)) { state="failed";reason="trajectory_numerical_or_allocation_error";code=70;break; }
        } else {
            int rc; size_t cut=0;
            rc=suffix?taxi_suffix_propose(&cur,&proposal,m,&rng,&cut):
                taxi_fixed_propose(&cur,&proposal,m,&rng,cdf,(size_t)max_steps);
            if (rc<0) { state="failed";reason="proposal_numerical_or_allocation_error";code=70;break; }
            if (rc>0) { accept=0;cost_changed=0;unfinished++; }
            else {
                cost_changed=proposal.sample.C!=cur.sample.C;
                double loga=taxi_log_accept(beta,cur.sample.C,proposal.sample.C,cur.sample.N,proposal.sample.N,suffix);
                if (isnan(loga)) { state="failed";reason="acceptance_numerical_error";code=70;break; }
                accept=log(taxi_uniform(&rng))<loga;
            }
            if (accept) { TaxiTrip temp=cur;cur=proposal;proposal=temp; }
        }
        attempts++;accepted+=accept;changed+=accept&&cost_changed;
        if (t<warmup) continue;
        unsigned char data[40+8*64]={0};
        u64le(data,t+1);
        if (conditional) {
            u64le(data+8,prefix.K);f64le(data+16,prefix.pre_cost);f64le(data+24,prefix.remainder);
            double cmean=taxi_prefix_mean(m,prefix);
            if (!isfinite(cmean)) { state="failed";reason="conditional_mean_overflow";code=70;break; }
            f64le(data+32,cmean);update(&cost,cmean);update(&steps,(double)prefix.K+1);
            for (size_t j=0;j<nt;j++) {
                double logp=taxi_log_tail(m,prefix,thresholds[j],0);
                double value=exp(logp);
                if (value==0 && isfinite(logp)) tail_underflows[j]++;
                f64le(data+40+8*j,logp);update(tails+j,value);
            }
        } else {
            f64le(data+8,cur.sample.C);f64le(data+16,cur.sample.X);
            u64le(data+24,cur.sample.N);u64le(data+32,cur.sample.long_steps);
            u64le(data+40,(uint64_t)accept | ((uint64_t)(accept&&cost_changed)<<1u));
            update(&cost,cur.sample.C);update(&steps,(double)cur.sample.N);update(&overshoot,cur.sample.X-m.L);
            atoms += m.b==0 || cur.sample.long_steps==0;
            for (size_t j=0;j<nt;j++) update(tails+j,cur.sample.C>=thresholds[j]);
        }
        if ((t-warmup)%stride==0) {
            int rc=record(&w,data,t+1);
            if (rc) {
                state=rc>0?"interrupted":"failed";reason=rc>0?"storage_limit":"trace_write_failure";code=rc>0?73:74;break;
            }
        }
    }
done: ;
    if (finish_chunk(&w)) { state="failed";reason="chunk_finalization_failure";code=74; }
    double wall=timer(CLOCK_MONOTONIC)-start, cpu=timer(CLOCK_PROCESS_CPUTIME_ID)-cpu_start;
    struct rusage usage; memset(&usage,0,sizeof(usage));(void)getrusage(RUSAGE_SELF,&usage);
    char summary[4096], temporary[4096];
    if (path_for(summary,sizeof(summary),w.dir,"engine-summary.json") ||
        path_for(temporary,sizeof(temporary),w.dir,"engine-summary.json.tmp")) code=74;
    else {
        FILE *f=fopen(temporary,"wx");
        if (!f) code=74;
        else {
            int rc=fprintf(f,"{\"schema_version\":1,\"status\":\"%s\",\"reason\":\"%s\",\"exit_code\":%d,"
                "\"iterations_completed\":%" PRIu64 ",\"post_warmup_count\":%" PRIu64
                ",\"retained_records\":%" PRIu64 ",\"finalized_chunks\":%" PRIu64 ",\"data_bytes\":%" PRIu64
                ",\"accepted\":%" PRIu64 ",\"accepted_cost_changes\":%" PRIu64 ",\"unfinished_proposals\":%" PRIu64
                ",\"atom_count\":%" PRIu64 ",\"mean_C\":%.17Lg,\"M2_C\":%.17Lg,\"mean_N\":%.17Lg,"
                "\"M2_N\":%.17Lg,\"mean_overshoot\":%.17Lg,\"M2_overshoot\":%.17Lg,"
                "\"cpu_seconds\":%.9g,\"wall_seconds\":%.9g,\"peak_rss_kib\":%ld,"
                "\"state_capacity_bytes\":%zu,\"rng_initial\":{\"state\":%" PRIu64 ",\"inc\":%" PRIu64 ",\"draws\":%" PRIu64
                "},\"rng_final\":{\"state\":%" PRIu64 ",\"inc\":%" PRIu64 ",\"draws\":%" PRIu64 "},\"tails\":[",
                state,reason,code,attempts,cost.n,w.published,w.chunks,w.used,accepted,changed,unfinished,atoms,
                cost.mean,cost.m2,steps.mean,steps.m2,overshoot.mean,overshoot.m2,cpu,wall,usage.ru_maxrss,
                (cur.capacity+proposal.capacity)*sizeof(double),initial.state,initial.inc,initial.draws,rng.state,rng.inc,rng.draws);
            for (size_t j=0;j<nt;j++)
                if (fprintf(f,"%s{\"threshold\":%.17g,\"mean\":%.17Lg,\"M2\":%.17Lg,\"linear_underflows\":%" PRIu64 "}",
                            j?",":"",thresholds[j],tails[j].mean,tails[j].m2,tail_underflows[j])<0) rc=-1;
            if (fputs("]}\n",f)==EOF) rc=-1;
            if (rc<0) { (void)fclose(f);code=74; }
            else if (durable_close(f) || rename(temporary,summary) || sync_directory(w.dir)) code=74;
        }
    }
    if (status_file(w.dir,code==0?"completed":(code==73||code==75||code>=128?"interrupted":"failed"),
                    attempts,w.published,wall)) code=74;
    taxi_free(&cur);taxi_free(&proposal);free(cdf);
    return code;
}
