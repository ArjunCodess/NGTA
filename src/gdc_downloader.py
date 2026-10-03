"""Local MAF reuse and reproducible, pinned public GDC acquisition."""
from pathlib import Path
import os

import pandas as pd

from .gdc_acquisition import pin_thca_manifest, download_pinned_manifest


def _find_existing_maf(download_dir: str | os.PathLike[str]) -> Path | None:
    root = Path(download_dir)
    for pattern in ("*.maf", "*tcga_mutations.tsv", "*mutations*.tsv"):
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def download_tcga_thca_maf(download_dir="./data", force_download=False):
    """Return local variants or acquire every file in an immutable GDC manifest.

    Local file reuse does not imply complete case or gene coverage. A new source
    version requires a new directory, since pinned manifests are never replaced.
    """
    root = Path(download_dir)
    root.mkdir(parents=True, exist_ok=True)
    existing = _find_existing_maf(root)
    manifest = root / "acquisition_manifest.json"
    if existing is not None and not force_download and not manifest.exists():
        print(f"Using local MAF {existing.name}; full assay coverage is not established")
        return pd.read_csv(existing, sep="\t", comment="#", low_memory=False)
    if not manifest.exists():
        pin_thca_manifest(manifest)
    report = download_pinned_manifest(manifest, root)
    frames = [pd.read_csv(root / record["file_name"], sep="\t", comment="#", low_memory=False)
              for record in report["files"]]
    if not frames:
        raise ValueError("Pinned GDC acquisition contains no eligible files")
    combined = pd.concat(frames, ignore_index=True)
    print(f"Verified {len(frames)} pinned files, {len(report['variant_case_ids'])} variant-bearing cases; gene callability remains unverified")
    return combined


def ensure_tcga_thca_maf(download_dir="./data"):
    existing = _find_existing_maf(download_dir)
    if existing is not None:
        return existing
    download_tcga_thca_maf(download_dir)
    existing = _find_existing_maf(download_dir)
    if existing is None:
        raise FileNotFoundError("Pinned GDC acquisition produced no MAF files")
    return existing


if __name__ == "__main__":
    print(download_tcga_thca_maf()["Hugo_Symbol"].value_counts().head(10))
