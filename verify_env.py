"""
verify_env.py — checks required env vars and internet connectivity to NEPSE data sources.
"""
import os
import sys
import io

# Force UTF-8 output on Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import requests
from dotenv import dotenv_values

# ── ANSI colours ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PASS = f"{GREEN}{BOLD} PASS {RESET}"
FAIL = f"{RED}{BOLD} FAIL {RESET}"
WARN = f"{YELLOW}{BOLD} WARN {RESET}"

# ── Config ───────────────────────────────────────────────────────────────────
REQUIRED_VARS = [
    "ANTHROPIC_API_KEY",
    "ZEP_API_KEY",
    "NEPSE_DB_URL",
    "NEWSAPI_KEY",
]

CONNECTIVITY_TARGETS = [
    ("nepse.com.np",      "https://nepse.com.np"),
    ("sharesansar.com",   "https://www.sharesansar.com"),
    ("merolagani.com",    "https://merolagani.com"),
]

TIMEOUT = 8  # seconds

# ── Helpers ──────────────────────────────────────────────────────────────────
def rule(char="-", width=60):
    return CYAN + char * width + RESET

def header(title):
    print(rule())
    print(f"{CYAN}|{RESET}  {BOLD}{title}{RESET}")
    print(rule())

def row(label, status, detail=""):
    label_col = f"{label:<30}"
    detail_col = f"  {YELLOW}{detail}{RESET}" if detail else ""
    print(f"  {label_col} [{status}]{detail_col}")

# ── Checks ───────────────────────────────────────────────────────────────────
def check_env_vars():
    header("Environment Variables (.env)")

    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        print(f"\n  {YELLOW}No .env file found — checking process environment only.{RESET}\n")
        values = {}
    else:
        values = dotenv_values(env_path)

    # Merge with actual process env so export-set vars also count
    merged = {**values, **os.environ}

    all_pass = True
    for var in REQUIRED_VARS:
        val = merged.get(var, "")
        is_placeholder = val in ("", "your_anthropic_api_key_here",
                                 "your_zep_api_key_here", "your_newsapi_key_here",
                                 "postgresql://user:password@localhost:5432/nepse_db")
        if not val:
            row(var, FAIL, "missing")
            all_pass = False
        elif is_placeholder:
            row(var, WARN, "placeholder value")
        else:
            row(var, PASS, f"{val[:6]}…")

    return all_pass

def check_connectivity():
    header("Internet Connectivity")

    all_pass = True
    for name, url in CONNECTIVITY_TARGETS:
        try:
            resp = requests.get(url, timeout=TIMEOUT,
                                headers={"User-Agent": "nepse-mirofish/1.0"})
            code = resp.status_code
            if code < 400:
                row(name, PASS, f"HTTP {code}")
            else:
                row(name, FAIL, f"HTTP {code}")
                all_pass = False
        except requests.exceptions.SSLError:
            # Site reachable but cert issue — treat as reachable
            row(name, WARN, "reachable (SSL warning)")
        except requests.exceptions.ConnectionError:
            row(name, FAIL, "connection refused / DNS failure")
            all_pass = False
        except requests.exceptions.Timeout:
            row(name, FAIL, f"timed out after {TIMEOUT}s")
            all_pass = False
        except Exception as e:
            row(name, FAIL, str(e)[:40])
            all_pass = False

    return all_pass

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print()
    env_ok  = check_env_vars()
    print()
    net_ok  = check_connectivity()
    print()
    print(rule("="))

    overall = env_ok and net_ok
    if overall:
        print(f"  {BOLD}Overall:{RESET} {GREEN}{BOLD}All checks passed.{RESET}")
    else:
        print(f"  {BOLD}Overall:{RESET} {RED}{BOLD}Some checks failed -- see above.{RESET}")

    print(rule("="))
    print()
    sys.exit(0 if overall else 1)

if __name__ == "__main__":
    main()
