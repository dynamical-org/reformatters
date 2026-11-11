# Cluster deployments

Each module in `src/reformatters/deployments/` defines its configured
`CLUSTER_DYNAMICAL_DATASETS` and `CLUSTER_OPERATIONAL_ARCHIVERS`.
`__main__.py` merges them into the global CLI registries. A resource must have
exactly one cluster owner. Move its configured instance between lists to change
ownership; storage configuration is independent of compute placement.

Cluster capacity and suspension policies live in `deployments/targets.py`.
Deploy applies the cluster policy after staging transforms. Cluster suspension
can only add suspension; it cannot unsuspend an individually suspended CronJob.
Backfill Jobs use the owning cluster's capacity type but are not suspended by
this scheduling policy. Ownership changes therefore require an operational
handoff, not just a registry edit.

## Identity prerequisite

Before using cluster submission commands, an administrator must install a
`reformatters-cluster-identity` ConfigMap in namespace `default`
on each target cluster, with `data.cluster` equal to its registry name, for
example `aws-us-west-2` or `aws-us-east-1`. The deployment and operator identities
need `get` on that named ConfigMap. They must not be able to alter its identity.
Workers do not need to read it.

The administrator should verify the deployment identity can read the ConfigMap
before the first deployment. A missing ConfigMap, denied read, or mismatched
identity stops submission. The preflight uses the current kubeconfig, just like
kubectl, and all submission commands use namespace `default`.

```sh
uv run main verify-cluster aws-us-west-2
uv run main resolve-cluster --dataset-id <dataset-id>
uv run main resolve-cluster --cronjob-name <cronjob-name>
uv run main deploy --cluster aws-us-west-2 --docker-image <image>
```

Backfill, staging deploy, and forced staging cleanup resolve ownership and check
identity before accessing storage or changing resources. An empty cluster's
deploy is a no-op. An explicit dataset filter belonging to another cluster fails.
The identity check does not fence older commits, previously created workloads,
or direct kubectl commands. Restrict old workflow/staging/rollback launch paths
and drain existing writers before moving ownership.

## Manual bootstrap

1. Create the target EKS Auto Mode cluster and ECR repository in the intended
   AWS account and region. Configure its VPC/subnets, NodeClass, storage classes,
   IAM access, and workload storage permissions. No infrastructure is created by
   these source files.
2. Review the region's NodePool manifest. For east, see
   [the NodePool notes](../deploy/aws/us-east-1/README.md). Review built-in and
   other NodePools so workloads cannot land on an unintended capacity pool.
3. Install the administrator-owned identity ConfigMap and grant its named read
   permission. Provision the required storage secrets and the same Sentry
   project configuration securely. Do not change output buckets or prefixes as
   part of compute placement.
4. Install and verify the trigger RBAC/admission prerequisites described in
   [the cluster setup documentation](cluster_setup.md).
5. Configure GitHub environment `prod-us-east-1` with `AWS_ROLE_TO_ASSUME`,
   `AWS_REGION`, `EKS_CLUSTER_NAME`, and `DOCKER_REPOSITORY`. Existing west uses
   `prod`. Confirm the publisher can read the source ECR image and write the
   target ECR image. Production images are built once and copied with their
   multi-architecture digest preserved.
6. Keep `AWS_US_EAST_1_DEPLOY_ENABLED` unset until bootstrap and target checks
   are complete. Enabling it allows east deployment; it does not remove the
   cluster's code-declared CronJob suspension policy. Keep the cluster suspended
   throughout preparation and validation.

Generic Jobs/Pods workflows take an explicit cluster. Dataset and CronJob
workflows resolve the cluster from the registry before selecting credentials.
Backfills wait for the selected cluster's successful deployment of the exact
commit, independently of another cluster's result.

Shared image metadata is retained for one day. After it expires, retry the whole
deployment workflow to regenerate it; an east-only retry cannot recover an
expired artifact. During a partial rerun that omits the selected cluster, the
backfill gate waits for the rerun to finish before using that cluster's earlier
successful result. Retry the backfill if its 30-minute wait expires.

## Moving a running workload

Prepare and validate the destination while its CronJobs are suspended. Arrange
an operator-owned window with checkable source, store, and active-job conditions.
A no-op update must finish validation with the same main snapshot; it does not
prove write permissions, polling behavior, or end-to-end publication latency.
Verify those separately before handing over writers.

Suspend and drain the source before removing its CronJobs and activating the
destination. Account for the activation commit's Code Quality and deployment
latency when reserving the window. If validation, CI, or deployment fails, leave
the destination suspended and recover source ownership/scheduling before its
next required update. Do not allow two clusters to write the same store during
rollback. Preserve Sentry monitor identities to retain history.

Server-side dry runs verify the submitting identity's RBAC,
schema, and applicable admission. They do not prove node fit, storage access, or
policies that apply only to a different service account.
