#!/usr/bin/env python3
"""Fail when Rust e2e test count drops below a minimum."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

_TEST_LINE_RE = re.compile(r"^\s*([^\s].*): test$")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check minimum number of Rust e2e tests."
    )
    parser.add_argument(
        "--min-tests",
        type=int,
        default=int(os.environ.get("E2E_MIN_TESTS", "3")),
        help="Minimum required number of discovered e2e tests.",
    )
    parser.add_argument(
        "--cargo",
        default=os.environ.get("CARGO", "cargo"),
        help="Cargo executable.",
    )
    parser.add_argument(
        "--locked",
        action="store_true",
        help="Pass --locked to cargo test.",
    )
    args = parser.parse_args()

    cmd = [args.cargo, "test"]  # nosemgrep: python.lang.security.audit.subprocess-run-audit.subprocess-run-audit
    if args.locked:
        cmd.append("--locked")
    cmd.extend(["--test", "e2e_tests", "--", "--list"])

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    output = f"{proc.stdout}\n{proc.stderr}"

    if proc.returncode != 0:
        print("Failed to list e2e tests.", file=sys.stderr)
        if output.strip():
            print(output, file=sys.stderr)
        return 2

    count = sum(1 for line in output.splitlines() if _TEST_LINE_RE.match(line))
    if count < args.min_tests:
        print(
            f"Discovered e2e tests: {count}. Minimum required: {args.min_tests}.",
            file=sys.stderr,
        )
        return 1

    print(f"Discovered e2e tests: {count}. Minimum required: {args.min_tests}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
