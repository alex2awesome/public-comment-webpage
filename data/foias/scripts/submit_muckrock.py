#!/usr/bin/env python3
"""
Submit FOIA requests to MuckRock and manage collaborators.

Usage:
    python submit_muckrock.py --dry-run                      # Preview submissions
    python submit_muckrock.py                                # Submit all 17
    python submit_muckrock.py --agency GSA                   # Submit one
    python submit_muckrock.py add-collaborators              # Add collaborators to all submitted requests
    python submit_muckrock.py add-collaborators --agency GSA  # Add to one

Auth:
    Submission uses Token auth from ~/.muckrock-api-key.txt
    Collaborator management uses JWT auth (prompts for MuckRock username/password,
    or reads from ~/.muckrock-credentials.json: {"username":"...","password":"..."})

Collaborators are configured in COLLABORATORS list below.
"""

import argparse
import getpass
import json
import sys
import time
from pathlib import Path

import requests as http_requests

# ── Configuration ──────────────────────────────────────────────────────

DRAFTS_DIR = Path(__file__).parent.parent / "tracking-spreadsheets" / "drafts"
API_V1 = "https://www.muckrock.com/api_v1"
API_V2 = "https://www.muckrock.com/api_v2"
AUTH_URL = "https://accounts.muckrock.com/api/token/"
REFRESH_URL = "https://accounts.muckrock.com/api/refresh/"

# People to add as collaborators on every request (MuckRock usernames)
# edit = can edit the request; read = view-only
COLLABORATORS = {
    "edit": [
        # "username1",
        # "username2",
    ],
    "read": [
        # "username3",
    ],
}

# Pre-resolved MuckRock agency IDs (federal jurisdiction=10)
AGENCIES = {
    "GSA":   (33,    "General Services Administration"),
    "HHS":   (74,    "Department of Health and Human Services"),
    "OPM":   (2381,  "Office of Personnel Management"),
    "AMS":   (5360,  "USDA Agricultural Marketing Service"),
    "CMS":   (239,   "Department of Health and Human Services, Centers for Medicare & Medicaid Services"),
    "DOJ":   (4203,  "Department of Justice"),
    "FCC":   (46,    "Federal Communications Commission"),
    "FEMA":  (284,   "Federal Emergency Management Agency"),
    "FWS":   (3805,  "U.S. Fish and Wildlife Service"),
    "NHTSA": (6031,  "National Highway Traffic Safety Administration"),
    "NSF":   (713,   "National Science Foundation"),
    "USCBP": (38,    "United States Customs and Border Protection"),
    "APHIS": (5196,  "U.S. Department of Agriculture, Animal Plant and Health Inspection Service"),
    "DHS":   (9,     "Department of Homeland Security"),
    "EPA":   (75,    "Environmental Protection Agency"),
    "FDA":   (116,   "Food and Drug Administration"),
    "FS":    (457,   "U.S. Forest Service"),
}

# ── Auth helpers ───────────────────────────────────────────────────────

def load_token_key():
    key_path = Path.home() / ".muckrock-api-key.txt"
    if not key_path.exists():
        print(f"ERROR: API key file not found at {key_path}")
        sys.exit(1)
    return key_path.read_text().strip()


def make_token_session(api_key):
    session = http_requests.Session()
    session.headers.update({
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
    })
    return session


def get_jwt_credentials():
    """Get MuckRock username/password from file or prompt."""
    cred_path = Path.home() / ".muckrock-credentials.json"
    if cred_path.exists():
        creds = json.loads(cred_path.read_text())
        return creds["username"], creds["password"]
    print("MuckRock JWT login required for collaborator management.")
    print(f"(To avoid this prompt, save credentials to {cred_path})")
    print(f'  Format: {{"username":"...","password":"..."}}')
    username = input("MuckRock username: ")
    password = getpass.getpass("MuckRock password: ")
    return username, password


def make_jwt_session(username, password):
    """Create a session with JWT Bearer auth."""
    resp = http_requests.post(AUTH_URL, json={
        "username": username,
        "password": password,
    })
    if resp.status_code != 200:
        print(f"ERROR: JWT login failed (HTTP {resp.status_code})")
        print(f"  {resp.text[:200]}")
        sys.exit(1)
    tokens = resp.json()
    session = http_requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {tokens['access']}",
        "Content-Type": "application/json",
    })
    session._refresh_token = tokens["refresh"]
    session._username = username
    session._password = password
    return session


def refresh_jwt(session):
    """Refresh an expired JWT access token."""
    resp = http_requests.post(REFRESH_URL, json={
        "refresh": session._refresh_token,
    })
    if resp.status_code == 200:
        tokens = resp.json()
        session.headers["Authorization"] = f"Bearer {tokens['access']}"
        if "refresh" in tokens:
            session._refresh_token = tokens["refresh"]
        return True
    # If refresh fails, re-login
    new_session = make_jwt_session(session._username, session._password)
    session.headers["Authorization"] = new_session.headers["Authorization"]
    session._refresh_token = new_session._refresh_token
    return True


def jwt_request(session, method, url, **kwargs):
    """Make a request with automatic JWT refresh on 401."""
    resp = getattr(session, method)(url, **kwargs)
    if resp.status_code == 401:
        refresh_jwt(session)
        resp = getattr(session, method)(url, **kwargs)
    return resp


# ── User lookup ────────────────────────────────────────────────────────

def resolve_user_ids(session, usernames):
    """Look up MuckRock user IDs from usernames using JWT auth."""
    user_ids = {}
    for username in usernames:
        resp = jwt_request(session, "get",
            f"{API_V2}/users/",
            params={"username": username, "format": "json"},
        )
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            if results:
                user_ids[username] = results[0]["id"]
            else:
                print(f"  WARNING: User '{username}' not found on MuckRock")
        else:
            print(f"  WARNING: Could not look up '{username}' (HTTP {resp.status_code})")
        time.sleep(0.3)
    return user_ids


# ── Submission helpers ─────────────────────────────────────────────────

def verify_agency(session, agency_abbrev):
    agency_id, agency_name = AGENCIES[agency_abbrev]
    resp = session.get(f"{API_V1}/agency/{agency_id}/", params={"format": "json"})
    if resp.status_code == 200:
        data = resp.json()
        return agency_id, data.get("name", agency_name)
    return agency_id, agency_name


def load_draft(agency_abbrev):
    path = DRAFTS_DIR / f"{agency_abbrev}.txt"
    if not path.exists():
        return None
    return path.read_text().strip()


def make_title(agency_abbrev):
    _, agency_name = AGENCIES[agency_abbrev]
    return f"AI Chatbot Interaction Logs — {agency_name}"


def load_submission_log():
    log_path = DRAFTS_DIR.parent.parent / "submission_log.json"
    if log_path.exists():
        return json.loads(log_path.read_text())
    return []


def save_submission_log(results):
    log_path = DRAFTS_DIR.parent.parent / "submission_log.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {log_path}")


# ── Commands ───────────────────────────────────────────────────────────

def cmd_submit(args):
    """Submit FOIA requests via v1 API with Token auth."""
    api_key = load_token_key()
    session = make_token_session(api_key)

    print("Testing authentication...")
    test_resp = session.get(f"{API_V1}/agency/33/", params={"format": "json"})
    if test_resp.status_code != 200:
        print(f"ERROR: Auth failed (HTTP {test_resp.status_code}).")
        sys.exit(1)
    print("  Auth OK")

    agencies_to_process = [args.agency.upper()] if args.agency else list(AGENCIES.keys())

    print("\nVerifying agency IDs...")
    agency_ids = {}
    for abbrev in agencies_to_process:
        agency_id, name = verify_agency(session, abbrev)
        agency_ids[abbrev] = (agency_id, name)
        print(f"  {abbrev}: {name} (id={agency_id})")
        time.sleep(0.3)

    print(f"\n{'='*60}")
    print(f"{'DRY RUN — ' if args.dry_run else ''}Submitting {len(agency_ids)} FOIA requests")
    print(f"{'='*60}\n")

    results = []
    for i, abbrev in enumerate(agencies_to_process, 1):
        draft = load_draft(abbrev)
        if not draft:
            print(f"[{i}/{len(agencies_to_process)}] {abbrev}: SKIPPED (no draft)")
            results.append({"agency": abbrev, "status": "skipped", "reason": "no draft"})
            continue

        agency_id, agency_name = agency_ids[abbrev]
        title = make_title(abbrev)

        payload = {
            "title": title,
            "full_text": draft,
            "agency": agency_id,
        }

        if args.dry_run:
            print(f"[{i}/{len(agencies_to_process)}] {abbrev}: WOULD SUBMIT")
            print(f"  Title:     {title}")
            print(f"  Agency:    {agency_name} (id={agency_id})")
            print(f"  Body:      {len(draft)} chars")
            print(f"  First 100: {draft[:100]}...")
            print()
            results.append({"agency": abbrev, "status": "dry_run", "title": title, "agency_id": agency_id})
        else:
            try:
                resp = session.post(f"{API_V1}/foia/", json=payload)
                if resp.status_code in (200, 201):
                    data = resp.json()
                    url = data.get("absolute_url", data.get("Location", "unknown"))
                    foia_ids = data.get("Requests", [])
                    print(f"[{i}/{len(agencies_to_process)}] {abbrev}: SUBMITTED")
                    print(f"  Request IDs: {foia_ids}")
                    print(f"  URL: https://www.muckrock.com{url}")
                    results.append({
                        "agency": abbrev,
                        "status": "submitted",
                        "request_ids": foia_ids,
                        "url": f"https://www.muckrock.com{url}",
                    })
                elif resp.status_code == 402:
                    print(f"[{i}/{len(agencies_to_process)}] {abbrev}: NO CREDITS")
                    results.append({"agency": abbrev, "status": "no_credits"})
                else:
                    print(f"[{i}/{len(agencies_to_process)}] {abbrev}: ERROR {resp.status_code}")
                    print(f"  {resp.text[:300]}")
                    results.append({"agency": abbrev, "status": "error", "code": resp.status_code})
            except Exception as e:
                print(f"[{i}/{len(agencies_to_process)}] {abbrev}: EXCEPTION — {e}")
                results.append({"agency": abbrev, "status": "exception", "error": str(e)})
            time.sleep(1.5)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    if args.dry_run:
        print(f"  Would submit: {sum(1 for r in results if r['status'] == 'dry_run')}")
    else:
        print(f"  Submitted: {sum(1 for r in results if r['status'] == 'submitted')}")
        print(f"  No credits: {sum(1 for r in results if r['status'] == 'no_credits')}")
        print(f"  Errors: {sum(1 for r in results if r['status'] in ('error','exception'))}")
    print(f"  Skipped: {sum(1 for r in results if r['status'] == 'skipped')}")

    save_submission_log(results)


def cmd_add_collaborators(args):
    """Add collaborators to submitted requests using JWT auth."""
    edit_usernames = COLLABORATORS["edit"]
    read_usernames = COLLABORATORS["read"]

    if not edit_usernames and not read_usernames:
        print("ERROR: No collaborators configured.")
        print("Edit the COLLABORATORS dict in this script to add MuckRock usernames.")
        sys.exit(1)

    # Load submission log to find request IDs
    log = load_submission_log()
    submitted = [r for r in log if r.get("status") == "submitted"]
    if not submitted:
        print("ERROR: No submitted requests found in submission_log.json.")
        print("Run 'python submit_muckrock.py' first to submit requests.")
        sys.exit(1)

    if args.agency:
        submitted = [r for r in submitted if r["agency"] == args.agency.upper()]
        if not submitted:
            print(f"ERROR: No submitted request found for {args.agency.upper()}")
            sys.exit(1)

    # JWT login
    username, password = get_jwt_credentials()
    session = make_jwt_session(username, password)
    print("JWT auth OK\n")

    # Resolve usernames to IDs
    all_usernames = list(set(edit_usernames + read_usernames))
    print(f"Resolving {len(all_usernames)} collaborator user IDs...")
    user_ids = resolve_user_ids(session, all_usernames)
    for uname, uid in user_ids.items():
        print(f"  {uname}: id={uid}")

    edit_ids = [user_ids[u] for u in edit_usernames if u in user_ids]
    read_ids = [user_ids[u] for u in read_usernames if u in user_ids]

    if not edit_ids and not read_ids:
        print("ERROR: Could not resolve any collaborator usernames.")
        sys.exit(1)

    # Add collaborators to each request
    print(f"\nAdding collaborators to {len(submitted)} requests...")
    print(f"  Edit access: {edit_usernames}")
    print(f"  Read access: {read_usernames}\n")

    for entry in submitted:
        abbrev = entry["agency"]
        request_ids = entry.get("request_ids", [])

        for req_id in request_ids:
            # First GET the current collaborators
            resp = jwt_request(session, "get",
                f"{API_V2}/requests/{req_id}/",
                params={"format": "json"},
            )
            if resp.status_code != 200:
                print(f"  {abbrev} (id={req_id}): GET failed ({resp.status_code})")
                continue

            current = resp.json()
            current_edit = current.get("edit_collaborators", [])
            current_read = current.get("read_collaborators", [])

            # Merge (don't duplicate)
            new_edit = list(set(current_edit + edit_ids))
            new_read = list(set(current_read + read_ids))

            if args.dry_run:
                print(f"  {abbrev} (id={req_id}): WOULD SET edit={new_edit} read={new_read}")
            else:
                patch_resp = jwt_request(session, "patch",
                    f"{API_V2}/requests/{req_id}/",
                    json={
                        "edit_collaborators": new_edit,
                        "read_collaborators": new_read,
                    },
                    params={"format": "json"},
                )
                if patch_resp.status_code in (200, 204):
                    print(f"  {abbrev} (id={req_id}): collaborators updated")
                else:
                    print(f"  {abbrev} (id={req_id}): PATCH failed ({patch_resp.status_code})")
                    print(f"    {patch_resp.text[:200]}")

            time.sleep(0.5)

    print("\nDone.")


def cmd_check_credits(args):
    """Check MuckRock credit balance for user and organization via JWT."""
    username, password = get_jwt_credentials()
    session = make_jwt_session(username, password)
    print("JWT auth OK\n")

    # Get user profile — /api_v2/users/me/ requires JWT
    resp = jwt_request(session, "get",
        f"{API_V2}/users/me/",
        params={"format": "json"},
    )
    if resp.status_code != 200:
        # Fall back to the accounts API for user profile
        print(f"users/me/ not available ({resp.status_code}), trying accounts API...")
        resp = jwt_request(session, "get",
            "https://accounts.muckrock.com/api/users/me/",
            params={"format": "json"},
        )

    if resp.status_code != 200:
        print(f"ERROR: Could not fetch user profile (HTTP {resp.status_code})")
        print(f"  {resp.text[:300]}")
        sys.exit(1)

    user = resp.json()
    print("=" * 60)
    print("USER")
    print("=" * 60)
    # Show known credit/balance fields
    for k in sorted(user.keys()):
        v = user[k]
        if any(term in k.lower() for term in ["credit", "balance", "quota", "request", "limit", "monthly", "remaining"]) or k in ("username", "id", "email", "full_name", "name"):
            print(f"  {k}: {v}")

    # List organizations the user belongs to
    orgs = user.get("organizations", []) or user.get("orgs", [])
    if orgs:
        print()
        print("=" * 60)
        print("ORGANIZATIONS")
        print("=" * 60)
        for org_ref in orgs:
            org_id = org_ref if isinstance(org_ref, int) else org_ref.get("id")
            if not org_id:
                continue
            # Try to get org details
            for url in [f"{API_V2}/organizations/{org_id}/",
                        f"https://accounts.muckrock.com/api/organizations/{org_id}/"]:
                org_resp = jwt_request(session, "get", url, params={"format": "json"})
                if org_resp.status_code == 200:
                    org = org_resp.json()
                    print(f"\n  Organization: {org.get('name','?')} (id={org_id})")
                    for k in sorted(org.keys()):
                        v = org[k]
                        if any(term in k.lower() for term in ["credit", "balance", "quota", "request", "limit", "monthly", "remaining", "member"]):
                            print(f"    {k}: {v}")
                    break
            else:
                print(f"  (could not fetch details for org {org_id})")


def cmd_list_org_requests(args):
    """List all FOIA requests under the Stanford AI-Usage FOIA organization."""
    username, password = get_jwt_credentials()
    session = make_jwt_session(username, password)
    print("JWT auth OK\n")

    # Find the org
    resp = jwt_request(session, "get",
        f"{API_V2}/users/me/",
        params={"format": "json"},
    )
    if resp.status_code != 200:
        resp = jwt_request(session, "get",
            "https://accounts.muckrock.com/api/users/me/",
            params={"format": "json"},
        )
    if resp.status_code != 200:
        print(f"ERROR: Could not fetch user profile (HTTP {resp.status_code})")
        sys.exit(1)

    user = resp.json()
    orgs = user.get("organizations", [])
    target_org_id = None
    for org_ref in orgs:
        org_id = org_ref if isinstance(org_ref, int) else org_ref.get("id")
        if not org_id:
            continue
        org_resp = jwt_request(session, "get",
            f"{API_V2}/organizations/{org_id}/",
            params={"format": "json"},
        )
        if org_resp.status_code == 200:
            org = org_resp.json()
            if org.get("name") == ORG_NAME:
                target_org_id = org_id
                break

    if not target_org_id:
        print(f"Could not find organization '{ORG_NAME}' via API.")
        print("Listing requests by known Stanford org member user IDs instead...")

    # Known Stanford AI-Usage FOIA org member user IDs (from recent activity)
    # Update this list as new members file requests
    STANFORD_ORG_USER_IDS = [83578, 182477, 182207]

    print(f"\nRequests by Stanford AI-Usage FOIA org members:")
    print("=" * 80)
    all_requests = []
    for uid in STANFORD_ORG_USER_IDS:
        resp = session.get(
            f"{API_V2}/requests/",
            params={"user": uid, "page_size": 100, "ordering": "-datetime_submitted", "format": "json"},
        )
        if resp.status_code == 200:
            data = resp.json()
            all_requests.extend(data.get("results", []))

    # Deduplicate and sort
    seen = set()
    unique = []
    for r in all_requests:
        if r["id"] not in seen:
            seen.add(r["id"])
            unique.append(r)
    unique.sort(key=lambda r: r.get("datetime_submitted") or "", reverse=True)

    # Group by status
    by_status = {}
    for r in unique:
        s = r.get("status", "?")
        by_status.setdefault(s, []).append(r)

    print(f"\nTotal: {len(unique)} requests")
    print("\nBy status:")
    for s, reqs in sorted(by_status.items(), key=lambda x: -len(x[1])):
        print(f"  {s:14s}: {len(reqs)}")

    # Flag withdrawn requests
    withdrawn = by_status.get("abandoned", [])
    if withdrawn:
        print(f"\n⚠️  WITHDRAWN/ABANDONED requests ({len(withdrawn)}):")
        for r in withdrawn:
            dt = (r.get("datetime_submitted") or "draft")[:10]
            print(f"  id={r['id']} user={r.get('user')} sub={dt} {r['title'][:65]}")

    print("\nAll requests:")
    for r in unique:
        dt = (r.get("datetime_submitted") or "draft")[:10]
        print(f"  id={r['id']} user={r.get('user')} status={r['status']:12s} sub={dt} {r['title'][:60]}")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Submit FOIA requests to MuckRock and manage collaborators",
    )
    parser.add_argument("command", nargs="?", default="submit",
                        choices=["submit", "add-collaborators", "check-credits", "list-org-requests"],
                        help="Command to run (default: submit)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without executing")
    parser.add_argument("--agency", type=str, help="Process only this agency (e.g., GSA)")
    args = parser.parse_args()

    if args.command == "submit":
        cmd_submit(args)
    elif args.command == "add-collaborators":
        cmd_add_collaborators(args)
    elif args.command == "check-credits":
        cmd_check_credits(args)
    elif args.command == "list-org-requests":
        cmd_list_org_requests(args)


if __name__ == "__main__":
    main()
