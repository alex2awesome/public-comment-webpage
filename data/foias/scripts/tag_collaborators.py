#!/usr/bin/env python3
"""
Tag all Stanford AI-Usage FOIA org members as collaborators on your MuckRock requests.

Each person must run this script themselves — MuckRock only allows you to
modify requests you own.

Usage:
    python tag_collaborators.py --dry-run    # Preview what would be tagged
    python tag_collaborators.py              # Tag everyone on everything

Setup:
    1. Get your MuckRock API token from https://www.muckrock.com/accounts/profile/
    2. Save it to ~/.muckrock-api-key.txt
    3. Run this script

What it does:
    - Finds all YOUR active (non-withdrawn) FOIA requests
    - Adds every other org member as an edit collaborator
    - Skips requests that already have everyone tagged
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests as http_requests

API_V1 = "https://www.muckrock.com/api_v1"
API_V2 = "https://www.muckrock.com/api_v2"

# Stanford AI-Usage FOIA org members
# Run with --list-members to refresh this from the API
ORG_MEMBERS = {
    115428: "Olivia Martin",
    182477: "Jacqueline Xiong",
    182459: "Nate Sanford",
    182399: "Tia Vasudeva",
    182207: "Raghav Sinha",
    182180: "Joachim Baumann",
    12759:  "Dan Bateyko",
    182178: "Vishakh Padmakumar",
    109303: "Alexander Spangher",
}


def load_api_key():
    key_path = Path.home() / ".muckrock-api-key.txt"
    if not key_path.exists():
        print(f"ERROR: API key not found at {key_path}")
        print("Get your token from https://www.muckrock.com/accounts/profile/")
        print(f"Then save it: echo 'YOUR_TOKEN' > {key_path}")
        sys.exit(1)
    return key_path.read_text().strip()


def get_my_user_id(session, headers):
    """Figure out who we are by checking a known request or searching."""
    # Try to find our user ID by making a test request to v1
    # The v1 API doesn't have a /me endpoint with Token auth,
    # so we search our own requests
    for uid, name in ORG_MEMBERS.items():
        r = session.get(f"{API_V2}/requests/",
            params={"user": uid, "page_size": 1, "format": "json"})
        if r.status_code == 200:
            results = r.json().get("results", [])
            # We can't tell from this if it's us — try PATCHing a request
            # Actually, let's just try each user and see which one we can PATCH
            pass

    # Alternative: try to find a request we own by testing PATCH
    # First get all requests from all org members
    for uid, name in ORG_MEMBERS.items():
        r = session.get(f"{API_V2}/requests/",
            headers=headers,
            params={"user": uid, "page_size": 1, "format": "json"})
        if r.status_code == 200:
            results = r.json().get("results", [])
            if results:
                test_id = results[0]["id"]
                # Try a no-op PATCH (set tags to current tags)
                pr = session.get(f"{API_V1}/foia/{test_id}/",
                    headers=headers,
                    params={"format": "json"})
                if pr.status_code == 200:
                    current_tags = pr.json().get("tags", [])
                    patch_r = session.patch(f"{API_V1}/foia/{test_id}/",
                        headers=headers,
                        json={"tags": current_tags},
                        params={"format": "json"})
                    if patch_r.status_code == 200:
                        print(f"  Identified as: {name} (id={uid})")
                        return uid
        time.sleep(0.2)

    print("ERROR: Could not identify your user ID.")
    print("You may not own any requests yet, or your API key may be wrong.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Tag all Stanford AI-Usage FOIA org members as collaborators on your requests",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without making changes")
    parser.add_argument("--list-members", action="store_true",
                        help="List all org members and exit")
    args = parser.parse_args()

    api_key = load_api_key()
    headers = {"Authorization": f"Token {api_key}", "Content-Type": "application/json"}
    session = http_requests.Session()

    if args.list_members:
        print("Stanford AI-Usage FOIA org members:")
        for uid, name in sorted(ORG_MEMBERS.items(), key=lambda x: x[1]):
            print(f"  id={uid:>6d}  {name}")
        return

    # Identify who we are
    print("Identifying your account...")
    my_id = get_my_user_id(session, headers)

    # Get all our active requests
    print(f"\nFetching your active requests...")
    r = session.get(f"{API_V2}/requests/",
        params={"user": my_id, "page_size": 100, "ordering": "-datetime_submitted", "format": "json"})
    if r.status_code != 200:
        print(f"ERROR: Could not fetch requests (HTTP {r.status_code})")
        sys.exit(1)

    all_requests = r.json().get("results", [])
    active = [req for req in all_requests if req.get("status") != "abandoned"]

    print(f"  Found {len(active)} active requests")

    if not active:
        print("No active requests to tag.")
        return

    # Other org members (everyone except me)
    others = {uid: name for uid, name in ORG_MEMBERS.items() if uid != my_id}
    other_ids = sorted(others.keys())

    print(f"\nWill add {len(others)} collaborators to each request:")
    for uid, name in sorted(others.items(), key=lambda x: x[1]):
        print(f"  {name} (id={uid})")

    # Process each request
    print(f"\n{'='*70}")
    print(f"{'DRY RUN — ' if args.dry_run else ''}Tagging {len(active)} requests")
    print(f"{'='*70}\n")

    tagged = 0
    skipped = 0
    errors = 0

    for i, req in enumerate(active, 1):
        rid = req["id"]
        title = req.get("title", "?")[:55]
        current_edit = req.get("edit_collaborators", [])

        # Check if everyone is already tagged
        missing = [uid for uid in other_ids if uid not in current_edit]

        if not missing:
            print(f"[{i}/{len(active)}] {title} — already tagged")
            skipped += 1
            continue

        new_edit = sorted(set(current_edit + other_ids))

        if args.dry_run:
            missing_names = [others[uid] for uid in missing if uid in others]
            print(f"[{i}/{len(active)}] {title}")
            print(f"  Would add: {', '.join(missing_names)}")
            tagged += 1
        else:
            r = session.patch(f"{API_V1}/foia/{rid}/",
                headers=headers,
                json={"edit_collaborators": new_edit},
                params={"format": "json"})

            if r.status_code == 200:
                missing_names = [others[uid] for uid in missing if uid in others]
                print(f"[{i}/{len(active)}] {title} — added {len(missing)}")
                tagged += 1
            else:
                print(f"[{i}/{len(active)}] {title} — ERROR {r.status_code}")
                errors += 1

            time.sleep(0.5)

    print(f"\n{'='*70}")
    print(f"Done: {tagged} tagged, {skipped} already complete, {errors} errors")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
