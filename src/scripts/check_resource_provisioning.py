"""
Check Kubernetes resource provisioning by comparing CloudWatch metrics to configured resources.

Analyzes actual CPU and memory usage (RSS) against configured requests.
Since we set request == limit, requests represent the OOM threshold.

Usage:
    uv run python src/scripts/check_resource_provisioning.py --log-group <log-group> [--days N] [--threshold PERCENT] [--namespace NAMESPACE]

Requirements:
    - AWS credentials configured (via ~/.aws/credentials or environment)
    - CloudWatch Container Insights enabled on the EKS cluster
"""

# ruff: noqa: T201 - allow prints in CLI script

import argparse
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import boto3
from botocore.client import BaseClient

# Import the dataset registry
from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common.kubernetes import CronJob


@dataclass
class ResourceConfig:
    """Configured resource requests for a job."""

    job_name: str
    dataset_id: str
    job_type: str  # "update" or "validate"
    cpu_requested: float  # cores
    memory_requested: float  # GB
    shared_memory_requested: float | None  # GB


@dataclass
class ResourceUsage:
    """Actual resource usage from CloudWatch."""

    job_name: str
    cpu_peak: float  # percent
    cpu_avg: float  # percent
    memory_peak: float  # GB
    memory_avg: float  # GB
    sample_count: int


@dataclass
class ProvisioningReport:
    """Report comparing configured vs actual resource usage.

    Since request == limit in our configuration, the request represents
    the OOM threshold for both CPU and memory.

    Memory calculation includes shared memory (/dev/shm) which counts against
    the container's memory limit in Kubernetes.
    """

    config: ResourceConfig
    usage: ResourceUsage | None
    cpu_overprovision_percent: float | None
    memory_overprovision_percent: float | None
    cpu_utilization_percent: float | None  # Peak CPU as % of request/limit
    memory_utilization_percent: float | None  # Peak total memory as % of request/limit
    total_memory_used: float | None  # Peak RSS + shared memory (GB)

    @property
    def is_overprovisioned(self) -> bool:
        """Check if CPU or memory is significantly overprovisioned."""
        if self.usage is None:
            return False
        # Consider overprovisioned if either resource is over threshold
        cpu_over = (
            self.cpu_overprovision_percent and self.cpu_overprovision_percent > 20
        )
        mem_over = (
            self.memory_overprovision_percent and self.memory_overprovision_percent > 20
        )
        return bool(cpu_over or mem_over)

    @property
    def is_underprovisioned(self) -> bool:
        """Check if resources are tight (>80% utilization)."""
        if self.usage is None:
            return False
        # Consider underprovisioned if using >80% of request/limit
        cpu_under = self.cpu_utilization_percent and self.cpu_utilization_percent > 80
        mem_under = (
            self.memory_utilization_percent and self.memory_utilization_percent > 80
        )
        return bool(cpu_under or mem_under)


def parse_cpu(cpu_str: str) -> float:
    """Parse CPU string (e.g., '7', '1.5', '500m') to cores."""
    if cpu_str.endswith("m"):
        return float(cpu_str[:-1]) / 1000
    return float(cpu_str)


def parse_memory(memory_str: str) -> float:
    """Parse memory string (e.g., '27G', '7Gi', '1024M') to GB."""
    memory_str = memory_str.strip()
    if memory_str.endswith("Gi"):
        return float(memory_str[:-2])
    elif memory_str.endswith("G"):
        return float(memory_str[:-1])
    elif memory_str.endswith("Mi"):
        return float(memory_str[:-2]) / 1024
    elif memory_str.endswith("M"):
        return float(memory_str[:-1]) / 1024
    else:
        raise ValueError(f"Unknown memory format: {memory_str}")


def get_configured_resources() -> list[ResourceConfig]:
    """Extract configured resources from all datasets."""
    configs = []

    for dataset in DYNAMICAL_DATASETS:
        try:
            # Get kubernetes resources for this dataset
            image_tag = (
                "placeholder:latest"  # Image tag doesn't affect resource configs
            )
            k8s_resources = dataset.operational_kubernetes_resources(image_tag)

            for resource in k8s_resources:
                if not isinstance(resource, CronJob):
                    continue

                job_type = "update" if "update" in resource.name else "validate"
                configs.append(
                    ResourceConfig(
                        job_name=resource.name,
                        dataset_id=dataset.dataset_id,
                        job_type=job_type,
                        cpu_requested=parse_cpu(resource.cpu),
                        memory_requested=parse_memory(resource.memory),
                        shared_memory_requested=(
                            parse_memory(resource.shared_memory)
                            if resource.shared_memory
                            else None
                        ),
                    )
                )
        except (AttributeError, TypeError, ValueError) as e:
            print(f"Warning: Failed to get resources for {dataset.dataset_id}: {e}")
            continue

    return configs


def query_cloudwatch_usage(
    job_name: str, days: int, log_group: str, namespace: str = "default"
) -> ResourceUsage | None:
    """Query CloudWatch Logs Insights for actual resource usage."""
    client = boto3.client("logs")

    end_time = datetime.now(tz=UTC)
    start_time = end_time - timedelta(days=days)

    # Query for CPU usage - parse JSON from @message
    cpu_query = f"""
    fields @timestamp, @message
    | parse @message /\"PodName\":\"(?<pod_name>[^\"]+)\"/
    | parse @message /\"namespace_name\":\"(?<k8s_namespace>[^\"]+)\"/
    | parse @message /\"Type\":\"(?<type>[^\"]+)\"/
    | parse @message /\"pod_cpu_utilization\":(?<cpu_util>[0-9.]+)/
    | filter k8s_namespace = "{namespace}"
    | filter pod_name like /{job_name}/
    | filter type = "Pod"
    | stats max(cpu_util) as peak_cpu_percent,
            avg(cpu_util) as avg_cpu_percent,
            count(*) as samples
    """

    # Query for memory usage - parse JSON from @message
    # Use RSS (Resident Set Size) instead of working_set to exclude page cache
    # Working set = RSS + cache, but cache can be evicted and doesn't cause OOM
    memory_query = f"""
    fields @timestamp, @message
    | parse @message /\"PodName\":\"(?<pod_name>[^\"]+)\"/
    | parse @message /\"namespace_name\":\"(?<k8s_namespace>[^\"]+)\"/
    | parse @message /\"Type\":\"(?<type>[^\"]+)\"/
    | parse @message /\"pod_memory_rss\":(?<mem_bytes>[0-9.]+)/
    | filter k8s_namespace = "{namespace}"
    | filter pod_name like /{job_name}/
    | filter type = "Pod"
    | stats max(mem_bytes) as peak_memory_bytes,
            avg(mem_bytes) as avg_memory_bytes,
            count(*) as samples
    """

    try:
        # Start CPU query
        cpu_response = client.start_query(
            logGroupName=log_group,
            startTime=int(start_time.timestamp()),
            endTime=int(end_time.timestamp()),
            queryString=cpu_query,
        )
        cpu_query_id = cpu_response["queryId"]

        # Start memory query
        memory_response = client.start_query(
            logGroupName=log_group,
            startTime=int(start_time.timestamp()),
            endTime=int(end_time.timestamp()),
            queryString=memory_query,
        )
        memory_query_id = memory_response["queryId"]

        # Wait for queries to complete
        cpu_results = wait_for_query(client, cpu_query_id)
        memory_results = wait_for_query(client, memory_query_id)

        if not cpu_results or not memory_results:
            return None

        # Parse results
        cpu_data = cpu_results[0]
        memory_data = memory_results[0]

        return ResourceUsage(
            job_name=job_name,
            cpu_peak=float(get_field(cpu_data, "peak_cpu_percent", "0")),
            cpu_avg=float(get_field(cpu_data, "avg_cpu_percent", "0")),
            memory_peak=float(get_field(memory_data, "peak_memory_bytes", "0"))
            / (1024**3),  # bytes to GB
            memory_avg=float(get_field(memory_data, "avg_memory_bytes", "0"))
            / (1024**3),
            sample_count=int(get_field(cpu_data, "samples", "0")),
        )

    except (RuntimeError, TimeoutError, KeyError, IndexError) as e:
        print(f"Warning: Failed to query CloudWatch for {job_name}: {e}")
        return None


def wait_for_query(
    client: BaseClient,
    query_id: str,
    timeout: int = 60,
) -> list[list[dict[str, str]]]:
    """Wait for CloudWatch query to complete and return results."""
    start = time.time()
    while time.time() - start < timeout:
        response = client.get_query_results(queryId=query_id)
        status = response["status"]

        if status == "Complete":
            return response["results"]
        elif status == "Failed":
            raise RuntimeError(f"Query {query_id} failed")

        time.sleep(1)

    raise TimeoutError(f"Query {query_id} timed out after {timeout}s")


def get_field(result: list[dict[str, str]], field_name: str, default: str = "0") -> str:
    """Extract field value from CloudWatch query result."""
    for item in result:
        if item["field"] == field_name:
            return item["value"]
    return default


def calculate_provisioning(
    config: ResourceConfig, usage: ResourceUsage | None
) -> ProvisioningReport:
    """Calculate provisioning metrics.

    Since request == limit, we compare usage against the request value,
    which represents the OOM threshold.

    Total memory = Peak RSS + shared_memory_requested
    Shared memory (/dev/shm with medium: Memory) counts against the container's
    memory limit in Kubernetes, so we must include it in utilization calculations.
    """
    if usage is None or usage.sample_count == 0:
        return ProvisioningReport(
            config=config,
            usage=usage,
            cpu_overprovision_percent=None,
            memory_overprovision_percent=None,
            cpu_utilization_percent=None,
            memory_utilization_percent=None,
            total_memory_used=None,
        )

    # CPU: peak utilization is already a percentage of the limit
    # Since request == limit, this tells us how close we are to the limit
    cpu_utilization = usage.cpu_peak
    cpu_peak_cores = config.cpu_requested * (usage.cpu_peak / 100)
    cpu_overprovision = (
        ((config.cpu_requested - cpu_peak_cores) / config.cpu_requested) * 100
        if cpu_peak_cores > 0
        else 0
    )

    # Memory: Total = Peak RSS + shared memory (if configured)
    # Shared memory counts against container memory limit in Kubernetes
    shared_mem = config.shared_memory_requested or 0
    total_memory_used = usage.memory_peak + shared_mem

    memory_utilization = (
        (total_memory_used / config.memory_requested) * 100
        if config.memory_requested > 0
        else 0
    )
    memory_overprovision = (
        ((config.memory_requested - total_memory_used) / config.memory_requested) * 100
        if total_memory_used > 0
        else 0
    )

    return ProvisioningReport(
        config=config,
        usage=usage,
        cpu_overprovision_percent=cpu_overprovision,
        memory_overprovision_percent=memory_overprovision,
        cpu_utilization_percent=cpu_utilization,
        memory_utilization_percent=memory_utilization,
        total_memory_used=total_memory_used,
    )


def print_report(reports: Iterable[ProvisioningReport], threshold: float) -> None:  # noqa: PLR0912, PLR0915
    """Print formatted resource provisioning report.

    Note: request == limit in our configuration, so utilization % is vs the OOM threshold.
    """
    print("\n" + "=" * 100)
    print("KUBERNETES RESOURCE PROVISIONING REPORT")
    print("=" * 100)
    print("Note: request == limit for all jobs (OOM occurs at request threshold)")
    print("=" * 100)

    reports_list = list(reports)
    overprovisioned = [r for r in reports_list if r.is_overprovisioned]
    underprovisioned = [r for r in reports_list if r.is_underprovisioned]
    no_data = [r for r in reports_list if r.usage is None or r.usage.sample_count == 0]
    ok = [
        r
        for r in reports_list
        if r.usage is not None
        and r.usage.sample_count > 0
        and not r.is_overprovisioned
        and not r.is_underprovisioned
    ]

    print("\nSummary:")
    print(f"  Total jobs analyzed: {len(reports_list)}")
    print(f"  Overprovisioned (>{threshold}% waste): {len(overprovisioned)}")
    print(f"  Underprovisioned (>80% utilization): {len(underprovisioned)}")
    print(f"  Appropriately sized: {len(ok)}")
    print(f"  No data available: {len(no_data)}")

    if underprovisioned:
        print(f"\n{'=' * 100}")
        print("⚠️  UNDERPROVISIONED JOBS (high risk of OOM)")
        print("=" * 100)

        for report in sorted(
            underprovisioned,
            key=lambda r: (r.memory_utilization_percent or 0),
            reverse=True,
        ):
            print(f"\nJob: {report.config.job_name}")
            print(f"  Dataset: {report.config.dataset_id}")
            print(f"  Type: {report.config.job_type}")
            print("\n  CPU:")
            print(
                f"    Request/Limit: {report.config.cpu_requested:.2f} cores (OOM threshold)"
            )
            if report.usage and report.cpu_utilization_percent:
                print(f"    Peak usage: {report.usage.cpu_peak:.1f}%")
                print(
                    f"    Utilization: {report.cpu_utilization_percent:.1f}% of limit"
                )
                if report.cpu_utilization_percent > 80:
                    print("    ⚠️  High CPU utilization!")
                recommended_cpu = (
                    report.config.cpu_requested * (report.usage.cpu_peak / 100) * 1.2
                )
                print(f"    Recommended: {recommended_cpu:.2f} cores")

            print("\n  Memory:")
            print(
                f"    Request/Limit: {report.config.memory_requested:.2f} GB (OOM threshold)"
            )
            if report.usage and report.memory_utilization_percent:
                print(f"    Peak RSS: {report.usage.memory_peak:.2f} GB")
                if report.config.shared_memory_requested:
                    print(
                        f"    Shared memory: {report.config.shared_memory_requested:.2f} GB (reserved)"
                    )
                if report.total_memory_used is not None:
                    print(
                        f"    Total used: {report.total_memory_used:.2f} GB (RSS + shared)"
                    )
                print(
                    f"    Utilization: {report.memory_utilization_percent:.1f}% of limit"
                )
                if report.memory_utilization_percent > 80:
                    print("    🔴 RISK: Close to OOM threshold!")
                if report.total_memory_used is not None:
                    recommended_memory = report.total_memory_used * 1.3
                    print(f"    Recommended: {recommended_memory:.2f} GB")

            if report.usage:
                print(f"\n  Samples: {report.usage.sample_count}")

    if overprovisioned:
        print(f"\n{'=' * 100}")
        print("OVERPROVISIONED JOBS (can reduce resources)")
        print("=" * 100)

        for report in sorted(
            overprovisioned,
            key=lambda r: (
                r.cpu_overprovision_percent or 0 + r.memory_overprovision_percent or 0
            ),
            reverse=True,
        ):
            print(f"\nJob: {report.config.job_name}")
            print(f"  Dataset: {report.config.dataset_id}")
            print(f"  Type: {report.config.job_type}")
            print("\n  CPU:")
            print(
                f"    Request/Limit: {report.config.cpu_requested:.2f} cores (OOM threshold)"
            )
            if report.usage:
                print(f"    Peak usage: {report.usage.cpu_peak:.1f}%")
                if report.cpu_utilization_percent:
                    print(
                        f"    Utilization: {report.cpu_utilization_percent:.1f}% of limit"
                    )
                print(
                    f"    Overprovision: {report.cpu_overprovision_percent:.1f}% 🔴"
                    if report.cpu_overprovision_percent
                    and report.cpu_overprovision_percent > threshold
                    else ""
                )
                recommended_cpu = (
                    report.config.cpu_requested * (report.usage.cpu_peak / 100) * 1.2
                )
                print(f"    Recommended: {recommended_cpu:.2f} cores")

            print("\n  Memory:")
            print(
                f"    Request/Limit: {report.config.memory_requested:.2f} GB (OOM threshold)"
            )
            if report.usage:
                print(f"    Peak RSS: {report.usage.memory_peak:.2f} GB")
                if report.config.shared_memory_requested:
                    print(
                        f"    Shared memory: {report.config.shared_memory_requested:.2f} GB (reserved)"
                    )
                if report.total_memory_used is not None:
                    print(
                        f"    Total used: {report.total_memory_used:.2f} GB (RSS + shared)"
                    )
                if report.memory_utilization_percent:
                    print(
                        f"    Utilization: {report.memory_utilization_percent:.1f}% of limit"
                    )
                if (
                    report.memory_overprovision_percent
                    and report.memory_overprovision_percent > threshold
                ):
                    print(
                        f"    Overprovision: {report.memory_overprovision_percent:.1f}% 🔴"
                    )
                if report.total_memory_used is not None:
                    recommended_memory = report.total_memory_used * 1.3
                    print(f"    Recommended: {recommended_memory:.2f} GB")

            if report.usage:
                print(f"\n  Samples: {report.usage.sample_count}")

    if ok:
        print(f"\n{'=' * 100}")
        print("✅ APPROPRIATELY SIZED JOBS (20-80% utilization)")
        print("=" * 100)
        for report in ok:
            print(f"\n{report.config.job_name}")
            print(
                f"  CPU: {report.config.cpu_requested:.2f} cores request/limit (OOM threshold)"
            )
            if report.usage and report.cpu_utilization_percent:
                print(
                    f"       {report.cpu_utilization_percent:.1f}% utilization ({report.usage.cpu_peak:.1f}% peak) ✓"
                )
            print(
                f"  Memory: {report.config.memory_requested:.2f} GB request/limit (OOM threshold)"
            )
            if (
                report.usage
                and report.memory_utilization_percent
                and report.total_memory_used
            ):
                shared_str = ""
                if report.config.shared_memory_requested:
                    shared_str = f" (RSS {report.usage.memory_peak:.2f} + shared {report.config.shared_memory_requested:.2f})"
                print(
                    f"          {report.memory_utilization_percent:.1f}% utilization ({report.total_memory_used:.2f} GB total{shared_str}) ✓"
                )

    if no_data:
        print(f"\n{'=' * 100}")
        print("NO DATA AVAILABLE")
        print("=" * 100)
        for report in no_data:
            print(f"  - {report.config.job_name}")
            print("    (Job may be suspended or not run in the selected time period)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check Kubernetes resource overprovisioning"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days of CloudWatch data to analyze (default: 7)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=20.0,
        help="Overprovisioning threshold percentage (default: 20)",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="default",
        help="Kubernetes namespace (default: default)",
    )
    parser.add_argument(
        "--log-group",
        type=str,
        required=True,
        help="CloudWatch log group name (e.g., /aws/containerinsights/<cluster-name>/performance)",
    )
    args = parser.parse_args()

    print(f"Analyzing resource usage over the last {args.days} days...")
    print(f"Overprovisioning threshold: {args.threshold}%")
    print(f"Namespace: {args.namespace}")
    print(f"Log group: {args.log_group}\n")

    # Get configured resources
    print("Gathering configured resources...")
    configs = get_configured_resources()
    print(f"Found {len(configs)} cron jobs\n")

    # Query CloudWatch for actual usage
    print("Querying CloudWatch Container Insights...")
    reports = []
    for config in configs:
        print(f"  Checking {config.job_name}...", end=" ")
        usage = query_cloudwatch_usage(
            config.job_name, args.days, args.log_group, args.namespace
        )
        if usage and usage.sample_count > 0:
            print(f"✓ ({usage.sample_count} samples)")
        else:
            print("⚠ No data")

        report = calculate_provisioning(config, usage)
        reports.append(report)

    # Print report
    print_report(reports, args.threshold)

    # Exit with error code if issues detected
    overprovisioned_count = sum(1 for r in reports if r.is_overprovisioned)
    underprovisioned_count = sum(1 for r in reports if r.is_underprovisioned)

    if underprovisioned_count > 0:
        print(
            f"\n🔴 Found {underprovisioned_count} underprovisioned jobs (high OOM risk)"
        )
        sys.exit(2)
    elif overprovisioned_count > 0:
        print(
            f"\n⚠️  Found {overprovisioned_count} overprovisioned jobs (cost optimization opportunity)"
        )
        sys.exit(1)
    else:
        print("\n✅ All jobs are appropriately sized")
        sys.exit(0)


if __name__ == "__main__":
    main()
