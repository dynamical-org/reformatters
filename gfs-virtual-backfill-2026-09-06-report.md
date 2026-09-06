# NOAA GFS forecast virtual backfill report

Job: `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf`

Dataset/store: `noaa-gfs-forecast-virtual`, `s3://dynamical-noaa-gfs/noaa-gfs-forecast-virtual/v0.1.0.icechunk`

## Outcome

The Job completed successfully. Kubernetes recorded `completedIndexes: 0-72`, 73 succeeded pods, zero failed pods, and an empty `failedIndexes`. All 1,460 expected six-hourly init positions in `[2021-05-01T00:00:00, 2022-05-01T00:00:00)` carry data. There were no OOMKilled containers, exit-137 containers, actual evictions, restarts, retried/duplicate indexes, errors, tracebacks, or log warnings in the successfully captured logs.

The Job ran from 2026-09-06 15:26:21Z to 15:51:44Z, or 25m23s. The local capture directory is:

`/tmp/gfs-virtual-backfill-2026-09-06t15-26-1-kyxf.AzjBuH`

That directory contains final Job and pod JSON, final namespace events, and 72 complete per-pod logs. The live attachment for index 70 raced container startup and Kubernetes garbage-collected that pod before the log could be recovered; the corresponding `.log` file contains only the failed recovery response. Sentry returned no rows for that exact pod. Kubernetes still recorded index 70 as `Succeeded`, exit 0, reason `Completed`, with a 165s wall time.

## Timing measurements

Percentiles use NumPy's default linear percentile calculation.

| Measurement | Count | Min | p50 | Mean | p95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pod container wall time | 73 | 137s | 170s | 180.0s | 235.2s | 571s |
| Commit latency | 72 of 73 | 6.3s | 26.5s | 37.7s | 88.4s | 418.3s |
| Reference emission | 72 of 73 | 25.8s | 28.9s | 28.8s | 31.3s | 32.0s |

Icechunk ancestry contains 73 `Update at ...` snapshots between repository initialization/expansion and Job completion, confirming the total worker commit count. Every one of the 72 captured commit lines reports 4,392,200 references. Thus the measured run wrote 73 worker commits covering approximately 320,630,600 references; only index 70's commit-latency line is unavailable.

The largest pod/commit outlier was index 12: 571s wall time and 418.3s commit latency. Its real lines were:

```text
[pod/noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-12-vmddd/worker] 2026-09-06T15:31:32Z INFO:reformatters.common.virtual_region_job:Ingesting 8360 files, 0 still pending (discover 13.8s, build 73.9s)
[pod/noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-12-vmddd/worker] 2026-09-06T15:39:21Z INFO:reformatters.common.virtual_region_job:Committed 4392200 refs (emit 29.7s, commit 418.3s)
```

## Memory substitute and termination audit

Every pod requested 7G memory and 3500m CPU. No pod had a memory limit. Direct memory use is unavailable because the cluster has no metrics-server and pod exec is denied, so no memory-use estimate is provided.

All 73 terminated containers had exit code 0 and reason `Completed`. Total restarts were zero. No pod status or final Job status contained OOMKilled, exit 137, or Evicted. The captured events contained no `NodeHasMemoryPressure`, `NodeHasDiskPressure`, or `NodeHasPIDPressure` event for the five nodes used by the Job; the observed node-condition events were `NodeHasSufficientMemory` and `NodeHasNoDiskPressure`.

There were transient scheduler events during initial capacity/PVC provisioning, including `Insufficient cpu`, `Insufficient memory`, and ephemeral-volume creation waits. These were scheduling-capacity messages, not container memory measurements or runtime failures. Three pods had `TaintManagerEviction` events whose message was `Cancelling deletion`; none was evicted and all completed with exit 0.

## Failure and log audit

- Restarts: 0.
- Indexes: 73 unique indexes, exactly 0 through 72; no missing, duplicate, or retried indexes.
- `failedIndexes`: empty.
- Errors/tracebacks/exceptions: none in the 72 successfully captured pod logs.
- Log-level warnings: none in the 72 successfully captured pod logs.
- `stale or mismatched index`: zero hits after explicitly grepping every captured `.log`. This covers 72/73 pods; index 70's application log was unavailable, so this is not presented as proof of a global zero.
- `Waiting for worker 0 to complete setup`: 2 lines, both during initial setup:

```text
/tmp/gfs-virtual-backfill-2026-09-06t15-26-1-kyxf.AzjBuH/noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-1-v88gc.log:2:[pod/noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-1-v88gc/worker] 2026-09-06T15:26:32Z INFO:reformatters.common.parallel_coordination:Waiting for worker 0 to complete setup...
/tmp/gfs-virtual-backfill-2026-09-06t15-26-1-kyxf.AzjBuH/noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-2-bdgf4.log:2:[pod/noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-2-bdgf4/worker] 2026-09-06T15:26:32Z INFO:reformatters.common.parallel_coordination:Waiting for worker 0 to complete setup...
```

## Final store state

The store was opened read-only with anonymous S3 storage plus anonymous authorization for `s3://noaa-gfs-bdp-pds/`. Its current `init_time` extent is 7,819 positions, as expected because no `append_dim_end` was supplied.

The requested year contains exactly 1,460 six-hourly coordinates, from 2021-05-01 00Z through 2022-04-30 18Z. Read-only Icechunk manifest existence checks for the representative `temperature_2m` lead-0 chunk at every one of those coordinates found 1,460/1,460 present, with no missing indexes. Positions after 2022-05-01 were excluded from this assessment as requested.

## Scaling comparison and full-scale sizing

| Measurement | Analysis reference | Forecast run | Forecast / analysis |
| --- | ---: | ---: | ---: |
| References per commit | 523,451 | 4,392,200 | 8.39x |
| Commit min | 0.6s | 6.3s | 10.50x |
| Commit p50 | 3.8s | 26.5s | 6.97x |
| Commit max | 14.9s | 418.3s | 28.07x |
| Pod wall min | 9s | 137s | 15.22x |
| Pod wall p50 | 13s | 170s | 13.08x |
| Pod wall mean | 13.6s | 180.0s | 13.23x |
| Pod wall p95 | 20s | 235.2s | 11.76x |
| Pod wall max | 25s | 571s | 22.84x |

Typical commit latency scaled slightly better than reference count: an 8.39x reference increase produced a 6.97x median commit increase. Tail behavior did not scale cleanly: the observed maximum was 28.07x the analysis maximum, and this tail drove pod time to roughly 13x at p50/mean and 22.84x at max. Sizing should therefore use the measured forecast throughput and p95/tail, not an 8.39x multiplier on the analysis run.

At parallelism 10, this run processed 1,460 init positions in 1,523s: 57.5 positions/minute or 3,451 positions/hour. For an arbitrary `N` positions at the same `jobs_per_pod=20`, the direct capacity inputs are:

- Pods/completions: `ceil(N / 20)`.
- Waves at parallelism 10: `ceil(ceil(N / 20) / 10)`.
- Concurrent requests: 35 CPU cores and 70G memory, with no memory limit.
- First-order elapsed projection from this run: `N / 57.5` minutes, subject to commit-tail and cluster-capacity variation.
- References: about 219,610 per init position.

For the current 7,819-position store extent, that is 391 pods, 40 waves, approximately 1.717 billion references, and a first-order elapsed projection of 135.9 minutes (2h16m) under the same cluster conditions. This is an elapsed-time projection only; memory cannot be sized from this run beyond reporting the configured 7G request because usage was not observable.

## Per-pod wall times and termination state

All rows have memory request 7G, CPU request 3500m, and no memory limit.

| Index | Pod | Wall | Exit | Reason | Restarts |
| ---: | --- | ---: | ---: | --- | ---: |
| 0 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-0-xs6hl` | 155s | 0 | Completed | 0 |
| 1 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-1-v88gc` | 167s | 0 | Completed | 0 |
| 2 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-2-bdgf4` | 261s | 0 | Completed | 0 |
| 3 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-3-99qbg` | 159s | 0 | Completed | 0 |
| 4 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-4-9c54g` | 184s | 0 | Completed | 0 |
| 5 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-5-gkd8f` | 170s | 0 | Completed | 0 |
| 6 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-6-fkcxs` | 252s | 0 | Completed | 0 |
| 7 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-7-7n9pq` | 190s | 0 | Completed | 0 |
| 8 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-8-pf4gj` | 148s | 0 | Completed | 0 |
| 9 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-9-fwcsk` | 204s | 0 | Completed | 0 |
| 10 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-10-hqdqx` | 224s | 0 | Completed | 0 |
| 11 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-11-lwpxv` | 152s | 0 | Completed | 0 |
| 12 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-12-vmddd` | 571s | 0 | Completed | 0 |
| 13 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-13-4h2vx` | 141s | 0 | Completed | 0 |
| 14 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-14-mvj8g` | 146s | 0 | Completed | 0 |
| 15 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-15-7xwkq` | 151s | 0 | Completed | 0 |
| 16 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-16-97w6j` | 174s | 0 | Completed | 0 |
| 17 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-17-86ltp` | 203s | 0 | Completed | 0 |
| 18 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-18-kv2dq` | 161s | 0 | Completed | 0 |
| 19 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-19-66lwv` | 181s | 0 | Completed | 0 |
| 20 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-20-6txvl` | 166s | 0 | Completed | 0 |
| 21 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-21-dgz7j` | 143s | 0 | Completed | 0 |
| 22 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-22-bvdjt` | 153s | 0 | Completed | 0 |
| 23 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-23-956wp` | 161s | 0 | Completed | 0 |
| 24 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-24-s29fn` | 162s | 0 | Completed | 0 |
| 25 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-25-hzbgl` | 198s | 0 | Completed | 0 |
| 26 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-26-4rptb` | 154s | 0 | Completed | 0 |
| 27 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-27-tvtx9` | 170s | 0 | Completed | 0 |
| 28 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-28-b7rvg` | 178s | 0 | Completed | 0 |
| 29 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-29-2cxk9` | 146s | 0 | Completed | 0 |
| 30 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-30-42dhv` | 202s | 0 | Completed | 0 |
| 31 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-31-g94ch` | 150s | 0 | Completed | 0 |
| 32 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-32-gpb46` | 143s | 0 | Completed | 0 |
| 33 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-33-z4pg9` | 166s | 0 | Completed | 0 |
| 34 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-34-gnbws` | 178s | 0 | Completed | 0 |
| 35 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-35-rjh2l` | 206s | 0 | Completed | 0 |
| 36 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-36-x8k6l` | 174s | 0 | Completed | 0 |
| 37 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-37-x6bbf` | 145s | 0 | Completed | 0 |
| 38 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-38-hd9tc` | 209s | 0 | Completed | 0 |
| 39 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-39-m45fp` | 149s | 0 | Completed | 0 |
| 40 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-40-dgvpt` | 145s | 0 | Completed | 0 |
| 41 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-41-2n8zv` | 180s | 0 | Completed | 0 |
| 42 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-42-7pqdt` | 181s | 0 | Completed | 0 |
| 43 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-43-mltmm` | 213s | 0 | Completed | 0 |
| 44 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-44-mc6mn` | 179s | 0 | Completed | 0 |
| 45 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-45-gf694` | 141s | 0 | Completed | 0 |
| 46 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-46-5xfqm` | 181s | 0 | Completed | 0 |
| 47 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-47-9dg54` | 212s | 0 | Completed | 0 |
| 48 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-48-24wtk` | 162s | 0 | Completed | 0 |
| 49 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-49-txprg` | 158s | 0 | Completed | 0 |
| 50 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-50-zz5fv` | 187s | 0 | Completed | 0 |
| 51 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-51-s64pp` | 191s | 0 | Completed | 0 |
| 52 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-52-c8547` | 176s | 0 | Completed | 0 |
| 53 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-53-9rcjr` | 166s | 0 | Completed | 0 |
| 54 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-54-72b2h` | 214s | 0 | Completed | 0 |
| 55 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-55-lflhp` | 141s | 0 | Completed | 0 |
| 56 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-56-xl9l8` | 223s | 0 | Completed | 0 |
| 57 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-57-kk8kg` | 144s | 0 | Completed | 0 |
| 58 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-58-htbrl` | 137s | 0 | Completed | 0 |
| 59 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-59-dmfxp` | 153s | 0 | Completed | 0 |
| 60 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-60-767c2` | 195s | 0 | Completed | 0 |
| 61 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-61-46tmk` | 198s | 0 | Completed | 0 |
| 62 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-62-mk6dz` | 265s | 0 | Completed | 0 |
| 63 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-63-b8x5s` | 142s | 0 | Completed | 0 |
| 64 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-64-5ctbd` | 195s | 0 | Completed | 0 |
| 65 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-65-jflqb` | 145s | 0 | Completed | 0 |
| 66 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-66-st9fm` | 138s | 0 | Completed | 0 |
| 67 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-67-ghm62` | 190s | 0 | Completed | 0 |
| 68 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-68-cvr7x` | 144s | 0 | Completed | 0 |
| 69 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-69-74nbj` | 188s | 0 | Completed | 0 |
| 70 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-70-zwkjw` | 165s | 0 | Completed | 0 |
| 71 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-71-f9mpd` | 175s | 0 | Completed | 0 |
| 72 | `noaa-gfs-forecast-vir-backfill-2026-09-06t15-26-1-kyxf-72-9tbjs` | 167s | 0 | Completed | 0 |
