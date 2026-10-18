# Cloud deployment

Use Docker images and Kubernetes indexed jobs for parallel processing. See [deploy/](../deploy/) and [deploy.py](../src/reformatters/common/deploy.py) for configuration and behavior.

## Setup

1. Install Docker and kubectl at `/usr/bin/docker` and `/usr/bin/kubectl`.
2. Configure a container registry, export `DOCKER_REPOSITORY`, and authenticate Docker for pushes.
3. Configure kubectl for the target cluster and create a Secret containing JSON storage options under the key `contents` (`kubectl create secret generic <name> --from-literal=contents='<json>'`); see [storage.py](../src/reformatters/common/storage.py).
4. As cluster administrator, install and verify trigger admission policies and authorize the deploy identity:

   ```sh
   uv run main render-kubernetes-admission-bundle --namespace default > admission.json
   kubectl apply -f admission.json -f deploy/trigger-binding-deployer.yaml
   uv run main verify-kubernetes-admission admission.json --namespace default
   kubectl create clusterrolebinding reformatters-trigger-binding-deployer \
     --clusterrole=reformatters-trigger-binding-deployer --user=DEPLOY_IDENTITY
   kubectl create rolebinding reformatters-trigger-binding-deployer --namespace default \
     --role=reformatters-trigger-binding-deployer --user=DEPLOY_IDENTITY
   ```

   The deploy identity also needs namespaced workload management and RBAC grants for every permission it delegates; external-authorizer grants do not satisfy Kubernetes escalation checks. Never grant wildcard verbs to the trigger account or its groups.

   Deploy creates bindings but never updates or deletes bindings or policies. It assumes policies remain installed and does not wait for admission propagation; a new target can be authorized before its binding is active on every API server. Before pruning bindings or replacing policies, pause deploys and let in-flight deploys finish. Revoke targets from the trigger Role and wait for propagation before pruning bindings; for policy replacement, revoke the trigger RoleBinding until verification and propagation complete.

5. Enable Restricted Pod Security warnings and audit:

   ```sh
   kubectl label namespace default --overwrite \
     pod-security.kubernetes.io/warn=restricted \
     pod-security.kubernetes.io/warn-version=v1.36 \
     pod-security.kubernetes.io/audit=restricted \
     pod-security.kubernetes.io/audit-version=v1.36
   ```

   Check every workload in the namespace, deploy compliant templates, verify new pods and storage writes, and let Jobs using old templates finish before enforcing:

   ```sh
   kubectl label namespace default --overwrite \
     pod-security.kubernetes.io/enforce=restricted \
     pod-security.kubernetes.io/enforce-version=v1.36
   ```

## Container updates

Pin container inputs to multi-platform digests. Dependabot handles base and tool images; update the Dockerfile frontend and Hatchling pins manually. Verify AMD64 and ARM64 builds when changing pins; Code Quality does not build runtime images.
