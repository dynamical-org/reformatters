# East Auto Mode NodePool

`nodepool.yaml` uses the cluster's `default` NodeClass and admits only on-demand,
ARM64 M8g instances, with generation greater than seven. The family restriction
is intentional: CPU and memory requests alone do not uniquely select M8g.
The NodeClass subnet selection determines the available AZs. Before applying,
review other NodePools (including built-ins) so they cannot provide a conflicting
capacity policy.

AWS lists `m8g.large` as 2 vCPU and 8 GiB in its
[general-purpose instance specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/gp.html).
It is the intended smallest fit for a 1.7 CPU / 7G pod, not a guaranteed rounding
result. `7G` is decimal bytes, not `7Gi`. The actual fit depends on node
allocatable resources and DaemonSet requests. Larger M8g sizes remain eligible
if overhead prevents a large from fitting. At bootstrap, check allocatable CPU
and memory plus DaemonSet reservations before expecting one workload per large.
A server-side dry run checks the manifest against the installed Auto Mode CRD;
it cannot validate capacity or scheduling fit.

## Availability zone evidence

No AZ is pinned: the public evidence does not establish which us-east-1 AZ has
the lowest latency to the actual R2 bucket. Cloudflare describes
[`enam` as Eastern North America](https://developers.cloudflare.com/r2/reference/data-location/)
and location hints as best effort. Its
[Snowflake region mapping](https://developers.cloudflare.com/r2/reference/partners/snowflake-regions/)
recommends `enam` for us-east-1, which supports the regional choice but does not
rank AZs. Confidence in selecting a particular AZ for R2 proximity is low.

For an empirical choice, an operator can run repeated small GETs and large range
GETs against the actual bucket endpoint from candidate AZs, using the same
instance type and network configuration. Compare TTFB p50/p95 and throughput
across multiple quiet windows. Use existing capacity where available; any new
measurement infrastructure requires the operator's own provisioning decision.
No such measurements have been run by this change. If differences are within
noise, choose for capacity and availability without claiming R2 proximity.

Record AZ IDs, not just account-dependent AZ names.
[AWS documents the mapping](https://docs.aws.amazon.com/ram/latest/userguide/working-with-az-ids.html),
and [Auto Mode documents `topology.kubernetes.io/zone`](https://docs.aws.amazon.com/eks/latest/userguide/create-node-pool.html)
for an eventual measured pin after resolving the ID to this account's zone name.

Network receive buffers (`net.core.rmem_max`, `net.ipv4.tcp_rmem`) and TCP
congestion-control tuning are planned follow-up work, requiring throughput
measurements and separate review. This manifest sets none of them.
