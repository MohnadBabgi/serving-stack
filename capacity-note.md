# Capacity note

## The numbers

- Locked model: `Qwen/Qwen2.5-1.5B-Instruct-AWQ`
- Target p95 end-to-end latency (your SLO today): `6.0` seconds
- Knee concurrency (highest concurrency whose p95 is still under target): `16` (sweep-bounded — p95 never crossed target within the tested range, so the true knee lies somewhere past 16)
- Tokens per second at the knee: `694.1`
- Max sustainable request rate at the target p95: `6.66` req/s

## The limiting family

- Compute-bound at this concurrency: throughput is still scaling near-linearly with load (88 → 694 tok/s from concurrency 1 to 16, roughly 7.9x for a 16x increase in concurrency) and p95 latency is climbing gradually rather than flattening or spiking, which is the tell that the GPU still has compute headroom left rather than hitting a memory-bandwidth or scheduling-overhead ceiling; the only early overhead signal is ttft_p95 nearly doubling from concurrency 8 to 16 (0.169s to 0.309s), suggesting queueing effects may start to matter before compute does at higher levels.

## Why the knee, not the peak

- The peak throughput number (694 tok/s at concurrency 16) only looks good because p95 latency at that level (2.404s) is still under my SLO by coincidence of where the sweep stopped; if I quoted throughput at an even higher concurrency without checking p95, I could be promising a number the server can only hit by making requests too slow to actually meet the target, so the knee — the highest concurrency that still respects the SLO — is the only number that reflects real, honest serving capacity.