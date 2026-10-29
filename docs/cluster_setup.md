## Deploying to the cloud

To reformat a large archive we parallelize work across multiple cloud servers.

We use
* `docker` to package the code and dependencies
* `kubernetes` indexed jobs to run work in parallel

Container image inputs in `deploy/Dockerfile` are pinned to multi-platform digests. Dependabot proposes base and tool image updates weekly, with a five-day cooldown. Update the Dockerfile frontend and Hatchling build-backend pins manually. Verify both AMD64 and ARM64 builds when updating pins; Code Quality does not build runtime images.

### Setup

1. Install `docker` and `kubectl`. Make sure `docker` can be found at `/usr/bin/docker` and `kubectl` at `/usr/bin/kubectl`.
1. Setup a docker image repository and export the `DOCKER_REPOSITORY` environment variable in your local shell. e.g. `export DOCKER_REPOSITORY=container.registry/<project-id>/reformatters/main`. Follow your registry's instructions to allow your docker to authenticate and push images to the registry.
1. Setup a kubernetes cluster and configure kubectl to point to your cluster. e.g. `aws eks update-kubeconfig --region <region> --name <cluster-name>`, `gcloud container clusters get-credentials <cluster-name> --region <region> --project <project>`, etc.
1. Create a kubectl secret containing a single json encoded value to be passed to fsspec `storage_options` or splatted as keyword arguments to an icechunk storage opener `kubectl create secret generic your-destination-storage-options-key --from-literal=contents='{"key": "...", "secret": "..."}'`. See `storage.py`.

1. As a cluster administrator, install the trigger admission policies and grant the deploy identity its RBAC roles once:

   ```sh
   uv run main render-kubernetes-admission-bundle --namespace default > admission.json
   kubectl apply -f admission.json -f deploy/trigger-binding-deployer.yaml
   uv run main verify-kubernetes-admission admission.json --namespace default
   kubectl create clusterrolebinding reformatters-trigger-binding-deployer \
     --clusterrole=reformatters-trigger-binding-deployer --group=DEPLOY_GROUP
   kubectl create rolebinding reformatters-trigger-binding-deployer --namespace default \
     --role=reformatters-trigger-binding-deployer --group=DEPLOY_GROUP
   ```

   The bootstrap grants every delegated verb through RBAC, because escalation checks ignore external authorizers; the deploy identity also needs ordinary workload permissions. Never grant wildcard verbs to `reformatters-update-trigger` or its groups. Dataset deploys need no further admin setup. Verification uses server-side dry runs and creates no resources; rerun it after the first deploy to probe each authorized CronJob.

   For policy changes, review `kubectl diff -f admission.json`, apply the bundle, and verify. Replace in place only when every changed object relaxes admission or leaves it unchanged; new policies or bindings count as tightening. Otherwise, replace with production and staging deploys paused, none in flight, and the trigger RoleBinding revoked across all API servers. Restore the RoleBinding and resume deploys only after verification passes and the policies are active on all API servers. Revocation blocks all trigger-account clones, including validation re-passes. Roll back by applying the previous bundle under the same rule.

   Prune bindings with deploys paused and none in flight, after removing their targets from the trigger Role and waiting for propagation.

1. Enable Restricted Pod Security warnings and audit before enforcement:

   ```sh
   kubectl label namespace default --overwrite \
     pod-security.kubernetes.io/warn=restricted \
     pod-security.kubernetes.io/warn-version=v1.36 \
     pod-security.kubernetes.io/audit=restricted \
     pod-security.kubernetes.io/audit-version=v1.36
   ```

   Check all workloads in the namespace, including those deployed elsewhere. Deploy the compliant templates, confirm replacement pods start and storage remains writable, and let Jobs using old templates finish. Old templates lacking the required security contexts will have replacement pods rejected. Once warnings are resolved, enforce:

   ```sh
   kubectl label namespace default --overwrite \
     pod-security.kubernetes.io/enforce=restricted \
     pod-security.kubernetes.io/enforce-version=v1.36
   ```
