import argparse
import csv
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run_command(command):
    process = subprocess.run(command, capture_output=True, text=True, shell=False, check=False)
    output = (process.stdout or "") + (process.stderr or "")
    return process.returncode, output


def docker_available():
    code, _ = run_command(["docker", "info"])
    return code == 0


def parse_locust_summary(output):
    data = {
        "requests": "",
        "failures": "",
        "avg_latency_ms": "",
        "p95_latency_ms": "",
        "rps": "",
    }

    req_match = re.search(
        r"Aggregated\s+(\d+)\s+(\d+)\([^)]*\)\s+\|\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\|\s+([\d.]+)\s+[\d.]+",
        output,
    )
    if req_match:
        data["requests"] = req_match.group(1)
        data["failures"] = req_match.group(2)
        data["avg_latency_ms"] = req_match.group(3)
        data["rps"] = req_match.group(4)

    lat_match = re.search(
        r"Aggregated\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\d+",
        output,
    )
    if lat_match:
        data["p95_latency_ms"] = lat_match.group(1)

    return data


def run_locust_once(host, users, spawn_rate, run_time):
    cmd = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        "locust/locustfile.py",
        "--host",
        host,
        "--headless",
        "-u",
        str(users),
        "-r",
        str(spawn_rate),
        "-t",
        run_time,
        "--only-summary",
    ]
    return run_command(cmd)


def main():
    parser = argparse.ArgumentParser(description="Run Locust benchmark for multiple API replica counts")
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--users", type=int, default=20)
    parser.add_argument("--spawn-rate", type=int, default=5)
    parser.add_argument("--run-time", default="20s")
    parser.add_argument("--replicas", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--skip-docker", action="store_true")
    args = parser.parse_args()

    evidence_dir = Path("evidence/flood-test")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    summary_path = evidence_dir / "summary.csv"

    can_scale = (not args.skip_docker) and docker_available()
    if not can_scale:
        print("Docker is not available (or skipped). Running Locust without changing replica count.")
    else:
        # Build once to avoid expensive rebuild on every replica iteration.
        base_up_cmd = ["docker", "compose", "up", "-d", "--build"]
        base_code, base_output = run_command(base_up_cmd)
        if base_code != 0:
            print(base_output)
            raise SystemExit("Failed to build/start docker services")

    rows = []
    for replicas in args.replicas:
        print(f"\n=== Running flood test for replicas={replicas} ===")

        if can_scale:
            scale_cmd = ["docker", "compose", "up", "-d", "--scale", f"api={replicas}"]
            scale_code, scale_output = run_command(scale_cmd)
            if scale_code != 0:
                print(scale_output)
                raise SystemExit(f"Failed to scale docker services for replicas={replicas}")

        code, output = run_locust_once(args.host, args.users, args.spawn_rate, args.run_time)
        log_path = evidence_dir / f"locust_replicas_{replicas}.log"
        log_path.write_text(output, encoding="utf-8")

        if code != 0:
            print(output)
            raise SystemExit(f"Locust failed for replicas={replicas}")

        parsed = parse_locust_summary(output)
        rows.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "replicas": replicas,
                "users": args.users,
                "spawn_rate": args.spawn_rate,
                "run_time": args.run_time,
                "requests": parsed["requests"],
                "failures": parsed["failures"],
                "avg_latency_ms": parsed["avg_latency_ms"],
                "p95_latency_ms": parsed["p95_latency_ms"],
                "rps": parsed["rps"],
                "log_file": str(log_path).replace("\\", "/"),
            }
        )

        print(f"Saved log: {log_path}")

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "replicas",
                "users",
                "spawn_rate",
                "run_time",
                "requests",
                "failures",
                "avg_latency_ms",
                "p95_latency_ms",
                "rps",
                "log_file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()