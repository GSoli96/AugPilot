#!/usr/bin/env python3
"""Recreate Algorithms/ by downloading the third-party repositories at a pinned commit.

Algorithms/ holds third-party code and is NOT part of this repository (see .gitignore). The manifest
scripts/algorithms_manifest.json lists, for each folder the code expects, the upstream URL, the pinned
commit, the licence and the local changes (patches and overlay files under scripts/).

For every entry the script
  1. skips it if Algorithms/<folder> already exists (an existing folder is never touched);
  2. clones the upstream repository (git clone --filter=blob:none --no-checkout) into a temporary folder
     next to the destination, checks out the pinned commit (or the default branch if there is none),
     optionally limited to a sparse path;
  3. applies the patches (git apply) and copies the overlay files;
  4. moves the result to Algorithms/<folder>.
Entries without a public URL, and files that cannot be restored automatically, are listed at the end as
manual actions.

Usage (from the project folder):
  python scripts/fetch_algorithms.py --dry-run            # show what would be done, no network
  python scripts/fetch_algorithms.py                      # fetch every non-optional entry
  python scripts/fetch_algorithms.py --only tab-ddpm-main TabDiff
  python scripts/fetch_algorithms.py --include-optional   # also the reference-only repositories
  python scripts/fetch_algorithms.py --dest some/folder   # download elsewhere (tests)
  python scripts/fetch_algorithms.py --no-git             # remove the .git folders after checkout

Notes: Python 3.8+ and git are needed. Repositories are cloned with core.autocrlf=false so that the patches
(LF line endings) apply on Windows too. Licences differ: read the "licence" field before reusing the code
(DIG is GPL-3.0, GraphAug is research-only, several repositories have no licence at all).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
DEFAULT_MANIFEST = SCRIPTS / "algorithms_manifest.json"
GIT = ["git", "-c", "core.autocrlf=false", "-c", "core.longpaths=true", "-c", "advice.detachedHead=false"]


def run(args, cwd=None, timeout=3600):
    """Run a git command; return (returncode, combined output)."""
    p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                       encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout + p.stderr).strip()


def rmtree(path):
    """Remove a folder tree also on Windows, where git object files are read-only."""
    def _onerror(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass
    if Path(path).exists():
        shutil.rmtree(path, onerror=_onerror)


def safe_name(name: str) -> bool:
    return bool(name) and name not in (".", "..") and "/" not in name and "\\" not in name and ":" not in name


def load_manifest(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data["entries"]
    for e in entries:
        if not safe_name(e["folder"]):
            sys.exit(f"Invalid folder name in the manifest: {e['folder']!r}")
    return entries


def select(entries, only, include_optional):
    if only:
        wanted = {w.lower() for chunk in only for w in chunk.replace(",", " ").split()}
        names = {e["folder"].lower(): e for e in entries}
        unknown = sorted(w for w in wanted if w not in names)
        if unknown:
            sys.exit("Unknown entries: %s\nAvailable: %s" % (", ".join(unknown), ", ".join(e["folder"] for e in entries)))
        return [e for e in entries if e["folder"].lower() in wanted]
    return [e for e in entries if include_optional or not e.get("optional")]


def describe(e):
    pin = e["commit"][:10] if e.get("commit") else f"no pin (ref {e.get('ref', 'HEAD')})"
    extra = []
    if e.get("sparse"):
        extra.append("sparse: " + ", ".join(e["sparse"]))
    if e.get("subdir"):
        extra.append("subdir: " + e["subdir"])
    if e.get("patches"):
        extra.append(f"{len(e['patches'])} patch")
    if e.get("overlay"):
        extra.append(f"{len(e['overlay'])} overlay file")
    return f"{e['url'] or '(no public URL)'} @ {pin}" + (" [" + "; ".join(extra) + "]" if extra else "")


def fetch_one(e, dest_root: Path, dry: bool, keep_git: bool):
    """Return (status, message)."""
    folder = e["folder"]
    target = dest_root / folder
    if target.exists():
        return "EXISTS", "already present, left untouched"
    if not e.get("url"):
        return "MANUAL", "no public URL"
    patches = [SCRIPTS / p for p in e.get("patches", [])]
    overlay = [(SCRIPTS / o["src"], o["dst"]) for o in e.get("overlay", [])]
    missing = [str(p) for p in patches + [s for s, _ in overlay] if not p.is_file()]
    if missing:
        return "FAILED", "missing patch/overlay files: " + ", ".join(missing)
    if dry:
        return "WOULD FETCH", describe(e)

    dest_root.mkdir(parents=True, exist_ok=True)
    work = dest_root / f".{folder}.partial"
    if work.exists():
        return "FAILED", f"leftover temporary folder {work} (remove it by hand and retry)"
    try:
        clone = work if not e.get("subdir") else work / "_repo"
        clone.parent.mkdir(parents=True, exist_ok=True)
        rc, out = run(GIT + ["clone", "--filter=blob:none", "--no-checkout", e["url"], str(clone)])
        if rc:
            raise RuntimeError("git clone failed: " + out[-400:])
        if e.get("sparse"):
            rc, out = run(GIT + ["sparse-checkout", "set", "--cone"] + e["sparse"], cwd=clone)
            if rc:
                raise RuntimeError("sparse-checkout failed: " + out[-400:])
        if e.get("commit"):
            rc, out = run(GIT + ["checkout", "--detach", e["commit"]], cwd=clone)
            if rc:  # commit not reachable from the default branch: fetch it explicitly
                rc2, out2 = run(GIT + ["fetch", "--filter=blob:none", "origin", e["commit"]], cwd=clone)
                if rc2:
                    raise RuntimeError("pinned commit not found: " + (out2 or out)[-400:])
                rc, out = run(GIT + ["checkout", "--detach", e["commit"]], cwd=clone)
        else:
            rc, out = run(GIT + ["checkout", e.get("ref") or "HEAD"], cwd=clone)
        if rc:
            raise RuntimeError("git checkout failed: " + out[-400:])

        if e.get("subdir"):  # keep only a subfolder of the repository as Algorithms/<folder>
            sub = clone / e["subdir"]
            if not sub.is_dir():
                raise RuntimeError(f"subdir {e['subdir']} not found in the repository")
            shutil.copytree(sub, work / "_out", ignore=shutil.ignore_patterns(".git"))
            rmtree(clone)
            for item in (work / "_out").iterdir():
                shutil.move(str(item), str(work / item.name))
            (work / "_out").rmdir()
            run(GIT + ["init", "-q"], cwd=work)  # patches are applied inside a repository of their own

        notes = []
        for p in patches:
            cwd = work
            flags = ["apply", "-p1", "--whitespace=nowarn"]
            rc, out = run(GIT + flags + ["--check", str(p)], cwd=cwd)
            if rc:  # line endings of the checkout may differ from those of the patch
                flags.append("--ignore-whitespace")
                rc, out = run(GIT + flags + ["--check", str(p)], cwd=cwd)
            if rc:
                raise RuntimeError(f"patch {p.name} does not apply: {out[-300:]}")
            rc, out = run(GIT + flags + [str(p)], cwd=cwd)
            if rc:
                raise RuntimeError(f"patch {p.name} failed: {out[-300:]}")
            notes.append(f"patch {p.name} applied" + (" (ignoring whitespace)" if len(flags) > 3 else ""))
        for src, rel in overlay:
            dst = work / rel
            if dst.exists():
                notes.append(f"overlay {rel} skipped (exists)")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            notes.append(f"overlay {rel} copied")
        if not keep_git:
            rmtree(work / ".git")
        work.rename(target)
        return "FETCHED", "; ".join([describe(e)] + notes)
    except Exception as exc:  # noqa: BLE001
        rmtree(work)
        return "FAILED", str(exc)
    finally:
        if work.exists() and not target.exists():
            rmtree(work)


def main():
    ap = argparse.ArgumentParser(description="Recreate Algorithms/ from the pinned upstream repositories.")
    ap.add_argument("--only", nargs="+", metavar="NAME", help="fetch only these folders (space or comma separated)")
    ap.add_argument("--dry-run", action="store_true", help="print what would be done; no network, no writes")
    ap.add_argument("--dest", default=str(ROOT / "Algorithms"), help="destination folder (default: Algorithms/)")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="manifest JSON (default: scripts/algorithms_manifest.json)")
    ap.add_argument("--include-optional", action="store_true", help="also fetch the reference-only (not executed) repositories")
    ap.add_argument("--no-git", action="store_true", help="delete the .git folder of each fetched repository")
    ap.add_argument("--list", action="store_true", help="list the manifest entries and exit")
    a = ap.parse_args()

    entries = load_manifest(Path(a.manifest))
    if a.list:
        for e in entries:
            print(f"{e['folder']:38} {e['role']:9} {'optional' if e.get('optional') else '        '} {describe(e)}  [{e['licence']}]")
        return 0
    if not a.dry_run and shutil.which("git") is None:
        sys.exit("git not found in PATH")

    chosen = select(entries, a.only, a.include_optional)
    dest_root = Path(a.dest).resolve()
    print(f"Destination: {dest_root}{'  (dry run)' if a.dry_run else ''}")
    print(f"Entries: {len(chosen)} of {len(entries)} in the manifest\n")

    results = []
    for e in chosen:
        status, msg = fetch_one(e, dest_root, a.dry_run, not a.no_git)
        results.append((e, status, msg))
        print(f"[{status:11}] {e['folder']}: {msg}", flush=True)

    counts = {}
    for _, s, _ in results:
        counts[s] = counts.get(s, 0) + 1
    print("\nSummary: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))

    manual = [(e, s) for e, s, _ in results if e.get("manual_action") and s != "EXISTS"]
    if manual:
        print("\nMANUAL ACTIONS REQUIRED")
        for e, s in manual:
            print(f"- {e['folder']}: {e['manual_action']}")
    mods = [e for e, s, _ in results if s != "EXISTS" and (e.get("patches") or e.get("overlay"))]
    if mods:
        print("\nProject changes to third-party code restored by patch/overlay (see the manifest for details):")
        for e in mods:
            print(f"- {e['folder']}: {e['local_modifications']}")
    lic = [e for e, s, _ in results if s != "EXISTS" and any(k in e["licence"].lower() for k in ("none", "gpl", "research", "cc by-nc"))]
    if lic:
        print("\nLicence warnings (do not redistribute these folders; check before reuse):")
        for e in lic:
            print(f"- {e['folder']}: {e['licence']}")
    return 1 if counts.get("FAILED") else 0


if __name__ == "__main__":
    sys.exit(main())
