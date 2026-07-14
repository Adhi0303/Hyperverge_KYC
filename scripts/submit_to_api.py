"""
scripts/submit_to_api.py
=========================
Submits pred.csv to the Hyperverge evaluation API.
"""
import requests
import os
import sys
import json

# ── CONFIG ────────────────────────────────────────────────
BASE_URL        = "http://3.108.8.61:8991"
HIRING_CODE     = "CIT_DE_2026"
NAME            = "ADARSH S"
EMAIL           = "suriyaadhi007@gmail.com"
PRED_CSV        = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "predictions", "pred.csv"))

# ── Validate ───────────────────────────────────────────────
if not os.path.exists(PRED_CSV):
    print(f"[ERROR] pred.csv not found at: {PRED_CSV}")
    sys.exit(1)

file_size_kb = os.path.getsize(PRED_CSV) / 1024
print("=" * 60)
print("  HyperVision KYC AI -- Submission Script")
print("=" * 60)
print(f"  Name        : {NAME}")
print(f"  Email       : {EMAIL}")
print(f"  Hiring Code : {HIRING_CODE}")
print(f"  CSV Path    : {PRED_CSV}")
print(f"  File Size   : {file_size_kb:.1f} KB")
print(f"  API URL     : {BASE_URL}/submit")
print("=" * 60)
print()

# ── Submit ─────────────────────────────────────────────────
def submit_predictions(pred_path, name, email, hiring_code):
    url = f"{BASE_URL}/submit"
    with open(pred_path, "rb") as f:
        files = {"predictions": (pred_path, f, "text/csv")}
        data  = {"name": name, "email": email, "hiring_code": hiring_code}
        print("[INFO] Sending request... (this may take 30-60 seconds)")
        resp  = requests.post(url, data=data, files=files, timeout=300)

    if not resp.ok:
        print(f"[FAIL] HTTP {resp.status_code}")
        try:
            print(resp.json().get("detail", resp.text))
        except ValueError:
            print(resp.text)
        return {}

    return resp.json()


def print_results(result):
    if not result:
        print("[INFO] No result data returned.")
        return
    print()
    print("=" * 60)
    print("  SUBMISSION RESULT")
    print("=" * 60)
    print(json.dumps(result, indent=2))
    print("=" * 60)


try:
    result = submit_predictions(PRED_CSV, NAME, EMAIL, HIRING_CODE)
    print_results(result)

    if result:
        print("\n[SUCCESS] Submission accepted! Check your scores above.")
    else:
        print("\n[WARN] Submission may have failed. Check the output above.")

except requests.exceptions.ConnectionError:
    print("\n[ERROR] Cannot reach the server. Check your internet connection.")
except requests.exceptions.Timeout:
    print("\n[ERROR] Request timed out after 5 minutes. Try again.")
except Exception as e:
    print(f"\n[ERROR] Unexpected error: {e}")
