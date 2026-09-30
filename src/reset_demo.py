"""
src/reset_demo.py

SIH26012 Demo State Reset & Cleanup Utility.
Safely removes demo-generated analysis runs while strictly protecting:
- Frozen model checkpoints
- Source raw datasets
- Research experiment directories and reports
- Phase 13, 14, 15, and 16 artifacts
- The reference prototype analysis directory (8b8c724c7e)

Usage:
    python src/reset_demo.py --dry-run
    python src/reset_demo.py --confirm
    python src/reset_demo.py --analysis-id <ID> --confirm
"""

import sys
import os
import shutil
import argparse
from pathlib import Path

# Safe terminal encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
API_STORAGE_ROOT = PROJECT_ROOT / "results/prototype/api"
PROTECTED_ANALYSIS_IDS = {"8b8c724c7e"}


def get_demo_analyses():
    """Identifies all analysis runs in results/prototype/api/ excluding protected ones."""
    if not API_STORAGE_ROOT.is_dir():
        return []

    candidates = []
    for item in API_STORAGE_ROOT.iterdir():
        if item.is_dir() and item.name not in PROTECTED_ANALYSIS_IDS:
            candidates.append(item)
    return sorted(candidates, key=lambda p: p.name)


def reset_demo(target_id: str = None, dry_run: bool = True, confirm: bool = False):
    print("=" * 70)
    print("  SIH26012 DEMO STATE RESET MECHANISM")
    print("=" * 70)
    print(f"Protected Reference Directory: results/prototype/api/8b8c724c7e/ [LOCKED]")
    print(f"Action Mode: {'DRY RUN (Preview Only)' if dry_run else 'EXECUTE REMOVAL'}\n")

    if target_id:
        if target_id in PROTECTED_ANALYSIS_IDS:
            print(f"[BLOCKED] Analysis ID '{target_id}' is permanently protected and cannot be deleted.")
            return False
        target_path = API_STORAGE_ROOT / target_id
        if not target_path.is_dir():
            print(f"[INFO] Analysis ID '{target_id}' does not exist in {API_STORAGE_ROOT}.")
            return True
        targets = [target_path]
    else:
        targets = get_demo_analyses()

    if not targets:
        print("[INFO] No demo-generated analysis directories found to clean.")
        return True

    print(f"Found {len(targets)} demo-generated analysis directories:")
    total_bytes = 0
    for t in targets:
        size = sum(f.stat().st_size for f in t.glob("**/*") if f.is_file())
        total_bytes += size
        print(f"  - {t.name:<16} ({size / (1024*1024):.2f} MB)")

    print(f"\nTotal reclaimable space: {total_bytes / (1024*1024):.2f} MB")

    if dry_run or not confirm:
        print("\n[NOTE] Dry run complete. No files were removed.")
        print("To execute removal, run:")
        print("    python src/reset_demo.py --confirm")
        return True

    print("\nExecuting removal of demo analysis directories...")
    removed_count = 0
    for t in targets:
        try:
            shutil.rmtree(t)
            removed_count += 1
            print(f"  [REMOVED] {t.name}")
        except Exception as e:
            print(f"  [ERROR] Failed to remove {t.name}: {e}")

    print(f"\n[OK] Successfully cleaned {removed_count} demo directories.")
    print("Protected reference analysis '8b8c724c7e' remains untouched.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIH26012 Demo Reset Utility")
    parser.add_argument("--dry-run", action="store_true", default=False, help="List targets without removing")
    parser.add_argument("--confirm", action="store_true", default=False, help="Confirm deletion")
    parser.add_argument("--analysis-id", type=str, default=None, help="Specific analysis ID to remove")

    args = parser.parse_args()

    # If neither dry-run nor confirm is explicitly given, default to dry-run for safety
    is_dry = args.dry_run or (not args.confirm)
    reset_demo(target_id=args.analysis_id, dry_run=is_dry, confirm=args.confirm)
