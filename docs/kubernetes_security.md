# Kubernetes workload security

The trigger ServiceAccount can read approved CronJobs and create/get Jobs. RBAC
cannot limit Job creation to a CronJob template. Admission policy supplies that
boundary: an identity-based guard requires an approved target, and a fixed-name
CronJob parameter binding compares the submitted Job to that target. Missing
parameters and evaluation errors deny the request.

The comparison covers the entire PodSpec, pod metadata, Job execution controls,
and Job labels/annotations. The exceptions are the server-generated Job selector
and controller labels, the selected CronJob name/UID labels, and the manual
instantiation annotation. A trigger cannot choose a different image, command,
service account, Secret reference, resource request, or additional container.
Unrecognized JobSpec fields are rejected until the policy is updated.

CronJob writers and admission-policy administrators remain trusted: changing a
template changes what the trigger may execute. A compromised trigger can repeat
approved Jobs and consume resources, and can use credentials already mounted in
its own pod. This policy is not a rate limit. Other identities, including the
CronJob controller and deployment identity, are outside its identity match.

## Admission installation and deployment

An operator with cluster-scoped administration rights must install the
`ValidatingAdmissionPolicy` and `ValidatingAdmissionPolicyBinding` resources.
Namespace Role permissions alone are insufficient. The ordinary deployment
identity needs its existing namespaced resource-management rights, plus CronJob
`get` and Job `create` for server-side dry runs; it needs neither impersonation
nor admission-policy write access. Treat anyone able to modify the policies or
their bindings as a security administrator.

Render from the revision being deployed, retaining every approved staging target:

```sh
uv run main render-admission-bundle --namespace default > admission.json
# Add --staging-target DATASET-VERSION-update for each approved staging version.
kubectl --context CONTEXT apply -f admission.json
KUBECONFIG=OPERATOR_CONFIG uv run main verify-admission
```

The bundle lists every registered `triggerable` CronJob. Each binding names one
CronJob parameter; an absent or suspended parameter denies cloning. The guard
rejects requests with no target label or an unapproved label, including requests
in another namespace using the same trigger identity.

Install and verify the bundle **before granting the trigger account any Job
creation permission**. When changing an already-active bundle, first stop trigger
workloads and revoke its Job-create grant, then apply and verify the complete
bundle before restoring the grant. Merely applying a binding before widening
the guard is insufficient: admission caches activate asynchronously. Do not
grant access while any policy reports compilation warnings or while verification
fails. Installation and grant restoration are operator actions, not automatic
policy deployment.

Ordinary deployment checks named-policy denials for an unlabeled Job and an
altered-image clone, and acceptance of an exact clone of each existing,
unsuspended target. These requests use `--dry-run=server` and an admission-canary
annotation; they create no Jobs or pods. The policies also match these dry-run
requests from the deployment identity, without giving it impersonation rights.
The annotation never exempts a real trigger request from comparison.

Only after the initial checks pass does deployment apply the ServiceAccount and
CronJobs. It checks the live templates again before applying the Role and
RoleBinding. Any failure stops later phases; existing objects are not rolled
back. New targets must be added to the operator bundle before deployment.
Staging deployments retain their normal trigger behavior but require explicit
staging target approval; production deploys do not overwrite that approval.

These smoke checks detect inactive or absent policies and bindings, not every
possible malicious policy edit. Review the installed policy configuration and
run the full local negative suite for policy changes. Admission administrators
and API-server exemptions are part of the trusted cluster configuration.

## Pod Security Admission rollout

Use an explicit context and namespace for every operator command. Namespace
labels affect **every** workload in the namespace, including workloads deployed
outside this repository. Do not enforce before inventorying those workloads.

1. Inventory CronJobs, Jobs, Pods, Deployments, StatefulSets, DaemonSets and
   ReplicaSets. Inspect every regular, init and ephemeral container, pod security
   context, host namespace setting and volume type. Record PSA exemptions too;
   labels cannot override API-server exemptions.
2. Enable Restricted warnings and audit at the tested Kubernetes version:

   ```sh
   kubectl --context CONTEXT label namespace NAMESPACE --overwrite \
     pod-security.kubernetes.io/warn=restricted \
     pod-security.kubernetes.io/warn-version=v1.36 \
     pod-security.kubernetes.io/audit=restricted \
     pod-security.kubernetes.io/audit-version=v1.36
   ```

3. Deploy compliant templates and inspect warnings, audit events and replacement
   pod startup. The common Job builder covers operational updates, validation,
   archivers and backfills. It runs as numeric UID/GID 999, requires non-root,
   uses RuntimeDefault seccomp, disables privilege escalation and drops all
   capabilities. The image uses the same numeric user. Secret/projected volumes,
   generic ephemeral PVCs and memory-backed `emptyDir` are permitted under
   Restricted. `fsGroup: 999` retains writable volume access; verify the storage
   driver's ownership behavior and actual application startup before enforcement.
4. Wait for Jobs created from older templates to finish. Updating a CronJob does
   not update its existing Jobs. Their replacement pods would still use the old
   security contexts and be rejected. Existing running pods are not evicted by
   adding PSA labels. Any template missing the four controls above must be fixed,
   as must host access, forbidden volumes, privileged containers or added
   capabilities reported by PSA.
5. Dry-run the enforcement label and resolve every warning, then enforce:

   ```sh
   kubectl --context CONTEXT label namespace NAMESPACE --overwrite \
     pod-security.kubernetes.io/enforce=restricted \
     pod-security.kubernetes.io/enforce-version=v1.36 --dry-run=server
   kubectl --context CONTEXT label namespace NAMESPACE --overwrite \
     pod-security.kubernetes.io/enforce=restricted \
     pod-security.kubernetes.io/enforce-version=v1.36
   ```

PSA rejects Pod creation; creation of a Job containing a noncompliant template
can still succeed with a warning. Test the resulting Pod template, not only the
Job API response. Keep warn/audit labels after enforcement and review version
pins when upgrading Kubernetes.

## Local admission tests

The admission test suite starts its own loopback-only API server and etcd, with
no kubelet or controllers and no production kubeconfig. CI downloads checksum-
verified Kubernetes 1.36 envtest binaries and runs it serially. To run locally,
point `REFORMATTERS_ADMISSION_TEST_ASSETS` at the directory containing
`kube-apiserver` and `etcd`, then run:

```sh
uv run pytest -n 0 tests/common/kubernetes_security_integration_test.py
```
