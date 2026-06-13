#!/usr/bin/env python3
"""Resolve the Trivy Linux-64bit tarball URL for a given release tag.

Usage: get_trivy_url.py <version-tag>
Prints the download URL to stdout.
Falls back to the latest release when the requested tag is not found.
"""
import json
import sys
import urllib.request
from urllib.error import HTTPError

requested_tag = sys.argv[1]
api_base = "https://api.github.com/repos/aquasecurity/trivy/releases"


def fetch(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.load(resp)


def pick_asset_url(release: dict) -> str | None:
    assets = release.get("assets", [])
    for asset in assets:
        name = asset.get("name", "")
        if name.endswith("_Linux-64bit.tar.gz"):
            return asset.get("browser_download_url")
    return None


release = None
try:
    release = fetch(f"{api_base}/tags/{requested_tag}")
except HTTPError:
    release = fetch(f"{api_base}/latest")

url = pick_asset_url(release)
if not url:
    raise SystemExit("Could not find Linux-64bit tarball for Trivy release")

print(url)
