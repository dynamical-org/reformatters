# Kubernetes workload security

The trigger ServiceAccount can read CronJobs in its namespace and create/get Jobs. RBAC
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
Names ending in a hyphen and decimal digits are reserved for scheduled Jobs;
trigger Jobs cannot occupy those names. Other Job names remain caller-selected.

CronJob writers and admission-policy administrators remain trusted: changing a
template changes what the trigger may execute. A compromised trigger can repeat
approved Jobs and consume resources, and can use credentials already mounted in
its own pod. This policy is not a rate limit. Other identities, including the
CronJob controller and deployment identity, are outside its identity match.

## Admission installation and deployment

An operator with cluster-scoped administration rights must install the
`ValidatingAdmissionPolicy` and `ValidatingAdmissionPolicyBinding` resources.
Namespace Role permissions alone are insufficient. The ordinary deployment
identity needs namespaced ServiceAccount/CronJob and Role/RoleBinding write
rights, plus CronJob `get` and Job `create/get`. Holding the granted permissions
also satisfies the RBAC anti-escalation checks for creating the Role and binding;
otherwise the corresponding `escalate`/`bind` authority is required. Job `create`
and CronJob `get` are also used for server-side dry runs. Deployment needs neither
impersonation nor admission-policy write access. See the
[RBAC grant restrictions](https://kubernetes.io/docs/reference/access-authn-authz/rbac/#privilege-escalation-prevention-and-bootstrapping). Treat anyone able to modify the policies or
their bindings as a security administrator.

Render from the revision being deployed, retaining every approved staging target:

```sh
uv run main render-admission-bundle --namespace default > admission.json
# Add --staging-target DATASET-VERSION-update for each approved staging version.
kubectl --context CONTEXT apply -f admission.json
KUBECONFIG=OPERATOR_CONFIG uv run main verify-admission admission.json --namespace default
```

Verification consumes the exact rendered bundle, including every staging approval,
and rejects an incomplete or edited bundle. The operator command reads every
installed policy and binding and requires its spec to equal the rendered spec,
including authenticated-identity matching. It checks missing parameters through
clone-policy denial, so a staging target can be approved before its CronJob exists.
Do not restore the create grant until all approved targets pass. Deploy applies
explicitly to `default`, matching the namespace its probes verify.

The bundle lists every registered `triggerable` CronJob. With no approved targets,
the identity guard denies every trigger Job. Workloads receive the account only
when explicitly configured to use it. Each binding names one
CronJob parameter; an absent or suspended parameter denies cloning. The guard
rejects requests with no target label or an unapproved label, including requests
in another namespace using the same trigger identity.

Install and verify the bundle **before granting the trigger account any Job
creation permission**. When changing an already-active bundle, first stop trigger
workloads, pause automated deployments, and revoke its Job-create grant, then
apply and verify the complete
bundle before restoring the grant. Merely applying a binding before widening
the guard is insufficient: admission caches activate asynchronously. Do not
grant access while any policy reports compilation warnings or while verification
fails. Installation and grant restoration are operator actions, not automatic
policy deployment. Resume deployments only after verification; deployment can
restore the RoleBinding.

Ordinary deployment checks named-policy denials for an unlabeled Job and an
altered-image clone, and acceptance of an exact clone of each existing,
unsuspended target. These requests use `--dry-run=server` and an admission-canary
annotation; they create no Jobs or pods. The policies also match these dry-run
requests from the deployment identity, without giving it impersonation rights.
The annotation never exempts a real trigger request from comparison.
Probe names are randomized. Exact-clone probes retry clone-policy denials for up
to 15 seconds while the parameter cache catches up with changed CronJobs;
remaining denials stop deployment. With no registered targets, both gates instead
require guard denial of an unapproved target. These probes do not prove that the
installed allowlist is empty; the operator's exact bundle comparison does.

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
The operator's spec comparison supplements the smoke tests; ordinary deployment
does not have cluster-scoped read access and uses only the smoke tests. In a
multi-replica control plane, successful probes through one endpoint do not prove
every replica's cache has converged; confirm convergence before restoring grants.
The identity guard is cluster-wide to reject this account in other namespaces.
An error in its match condition would have a cluster-wide Job-creation impact;
non-trigger and cross-namespace requests are included in the local API tests.

## Pod Security Admission rollout

Use an explicit context and namespace for every operator command. Namespace
labels affect **every** workload in the namespace, including workloads deployed
outside this repository. Do not enforce before inventorying those workloads.

1. Inventory CronJobs, Jobs, Pods, Deployments, StatefulSets, DaemonSets and
   ReplicaSets. Inspect every regular, init and ephemeral container, pod security
   context, host namespace setting and volume type. Record PSA exemptions too;
   labels cannot override API-server exemptions.
2. Confirm the server minor version with `kubectl --context CONTEXT version`.
   Enable Restricted warnings and audit at the tested server version (v1.36 here):

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
   Retire obsolete validation CronJobs or update their templates too; otherwise
   they keep creating Jobs with old security contexts after current Jobs drain.
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

An API upgrade that introduces a defaulted JobSpec field can make exact clones
fail the field allowlist. Review the new field, update the generator and run the
admission suite against that API version before re-rendering the operator bundle.
Check controller labels/selectors and Job defaults as well as the field list.
Do not remove the allowlist or switch bindings to Warn to restore triggering.
Scheduled CronJob-controller Jobs remain outside the trigger-identity policy.

## Local admission tests

The admission test suite starts its own loopback-only API server and etcd, with
no kubelet or controllers and no production kubeconfig. CI downloads checksum-
verified Kubernetes 1.36 envtest binaries and runs it serially. The module is
marked `slow`, so `-m "not slow"` excludes it even when assets are configured. To run locally,
point `REFORMATTERS_ADMISSION_TEST_ASSETS` at the directory containing
`kube-apiserver` and `etcd`, then run:

```sh
uv run pytest -n 0 tests/common/kubernetes_security_integration_test.py
```
