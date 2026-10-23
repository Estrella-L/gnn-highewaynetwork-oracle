"""Download and verify the archived terrain inputs; does not start training."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://github.com/Estrella-L/gnn-highewaynetwork-oracle/releases/download/terrain-datasets-20261023/"

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    manifest = json.loads((ROOT / "datasets/manifest.json").read_text())
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=list(manifest["datasets"]))
    ap.add_argument("--archive-dir", type=Path, help="Use already-downloaded gzip assets")
    ap.add_argument("--data-root", type=Path, default=ROOT / "datasets")
    args = ap.parse_args()
    names = [args.dataset] if args.dataset else list(manifest["datasets"])
    for name in names:
        record = manifest["datasets"][name]
        directory = args.data_root / name
        directory.mkdir(parents=True, exist_ok=True)
        out = directory / record["data_asset"][:-3]
        if out.exists():
            if sha256(out) != record["data_sha256"]:
                raise RuntimeError(f"Existing file has a different hash: {out}")
            print(f"Already verified: {out}")
            continue
        archive = (args.archive_dir or directory) / record["data_asset"]
        if not archive.exists():
            if args.archive_dir:
                raise FileNotFoundError(archive)
            download_tmp = archive.with_suffix(archive.suffix + ".download")
            with urlopen(BASE + record["data_asset"], timeout=120) as response, download_tmp.open("wb") as f:
                shutil.copyfileobj(response, f)
            download_tmp.replace(archive)
        tmp = out.with_suffix(out.suffix + ".tmp")
        try:
            with gzip.open(archive, "rb") as source, tmp.open("wb") as target:
                shutil.copyfileobj(source, target)
            if tmp.stat().st_size != record["data_bytes"] or sha256(tmp) != record["data_sha256"]:
                raise RuntimeError(f"Terrain integrity check failed: {name}")
            tmp.replace(out)
        finally:
            if tmp.exists():
                tmp.unlink()
        print(f"Verified: {out}")

if __name__ == "__main__":
    main()
