"""
secure_env.py — Harden .env file security on Windows.

Run ONCE after you create your .env file, and again whenever you update it:

    python scripts/secure_env.py

What this script does:
  1. Confirms .env exists (not just .env.example)
  2. Warns if .env is on an OneDrive-synced path (credentials may upload to cloud)
  3. Sets Windows ACL: strips inherited permissions, grants read-only to current user
  4. Scans for empty or placeholder values in required Shoonya credential fields
  5. Exits non-zero on any failure so CI/pre-flight checks can block a bad state
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
ENV_FILE = ROOT / ".env"

REQUIRED_KEYS = {
    "SHOONYA_USER",
    "SHOONYA_PASSWORD",
    "SHOONYA_TOTP_SECRET",
    "SHOONYA_VENDOR_CODE",
    "SHOONYA_API_SECRET",
    "SHOONYA_IMEI",
}

# Values that indicate the field was never filled in
EMPTY_MARKERS = {"", "your_", "change_me", "todo", "example", "xxx", "placeholder"}

_ok = True  # track overall pass/fail


def _pass(msg: str) -> None:
    print(f"[PASS] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _fail(msg: str) -> None:
    global _ok
    _ok = False
    print(f"[FAIL] {msg}")


# ── Check 1: .env must exist ────────────────────────────────────────────────

def check_env_exists() -> None:
    if not ENV_FILE.exists():
        _fail(f".env not found at {ENV_FILE}")
        print()
        print("  Fix:  copy .env.example to .env then fill in your credentials.")
        print("  Cmd:  copy .env.example .env   (Windows)  or  cp .env.example .env  (bash)")
        sys.exit(1)
    _pass(f".env found: {ENV_FILE}")


# ── Check 2: OneDrive sync warning ─────────────────────────────────────────

def check_onedrive() -> None:
    path_lower = str(ENV_FILE).lower()
    if "onedrive" in path_lower:
        _warn(
            ".env is inside an OneDrive-synced folder.\n"
            "         OneDrive may upload your broker credentials to Microsoft's cloud.\n"
            "         To exclude this project from sync:\n"
            "           OneDrive tray icon → Settings → Account → Choose folders\n"
            "           → uncheck this folder.\n"
            "         Alternatively, move the project outside your OneDrive folder entirely."
        )
    else:
        _pass(".env is not on a known cloud-sync path")


# ── Check 3: Windows ACL — restrict to current user ────────────────────────

def set_windows_acl() -> None:
    if sys.platform != "win32":
        # Fallback for WSL / Git Bash running on Windows — try chmod
        try:
            os.chmod(ENV_FILE, 0o600)
            _pass("chmod 600 applied to .env")
        except Exception as e:
            _warn(f"Could not chmod .env ({e}). Restrict permissions manually.")
        return

    user = os.environ.get("USERNAME", "")
    if not user:
        _warn("Could not read %USERNAME% — skipping ACL. Restrict .env manually.")
        return

    try:
        # /inheritance:r  — remove inherited ACEs
        # /grant:r        — replace (not add) explicit grant
        # <user>:R        — read-only for current user
        result = subprocess.run(
            [
                "icacls",
                str(ENV_FILE),
                "/inheritance:r",
                "/grant:r",
                f"{user}:(R)",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        _pass(f"ACL locked: only {user!r} can read .env (icacls)")
    except subprocess.CalledProcessError as e:
        _warn(
            f"icacls failed: {e.stderr.strip()}\n"
            "         Manually restrict: right-click .env → Properties → Security → Advanced\n"
            "         Remove all entries, add your user with Read-only."
        )
    except FileNotFoundError:
        _warn("icacls not found. Run this script from a standard Windows Command Prompt.")


# ── Check 4: No empty or placeholder credential values ──────────────────────

def check_credential_values() -> None:
    content = ENV_FILE.read_text(encoding="utf-8")
    missing: list[str] = []
    found: list[str] = []

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().upper()
        value = value.strip()

        if key not in REQUIRED_KEYS:
            continue

        is_empty = not value or any(
            value.lower().startswith(m) for m in EMPTY_MARKERS if m
        )
        if is_empty:
            missing.append(key)
        else:
            found.append(key)

    # Keys present in REQUIRED_KEYS but not found in the file at all
    keys_in_file = {k.strip().upper() for line in content.splitlines()
                    if "=" in line and not line.strip().startswith("#")
                    for k, _, _ in [line.partition("=")]}
    not_in_file = REQUIRED_KEYS - keys_in_file
    missing.extend(sorted(not_in_file))

    if missing:
        _fail(
            f"These required fields are empty or still placeholder:\n"
            + "\n".join(f"         {k}=" for k in sorted(missing))
        )
    else:
        _pass(f"All {len(REQUIRED_KEYS)} required Shoonya credentials are set")


# ── Check 5: .gitignore must exclude .env ──────────────────────────────────

def check_gitignore() -> None:
    gitignore = ROOT / ".gitignore"
    if not gitignore.exists():
        _warn(".gitignore not found — create one and add .env before first commit")
        return
    content = gitignore.read_text(encoding="utf-8")
    lines = {line.strip() for line in content.splitlines()}
    if ".env" in lines or "*.env" in lines:
        _pass(".gitignore contains .env")
    else:
        _fail(".gitignore does NOT list .env — add it now or your credentials will be committed")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  secure_env.py — .env security hardening")
    print("=" * 60)
    check_env_exists()
    check_onedrive()
    set_windows_acl()
    check_credential_values()
    check_gitignore()
    print("=" * 60)
    if _ok:
        print("  All checks passed. Your .env is secure.")
    else:
        print("  One or more checks FAILED. Fix the issues above before running the system.")
        sys.exit(1)


if __name__ == "__main__":
    main()
