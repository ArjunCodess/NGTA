import json
import os
import gzip
import shutil
from pathlib import Path

import pandas as pd
import requests

def _find_existing_maf(download_dir: str | os.PathLike[str]) -> Path | None:
    base_dir = Path(download_dir)
    for pattern in ("*.maf", "*tcga_mutations.tsv", "*mutations*.tsv"):
        matches = sorted(base_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def download_tcga_thca_maf(download_dir: str | os.PathLike[str] = "./data", force_download: bool = False) -> pd.DataFrame:
    """
    Queries the GDC API for the TCGA-THCA MAF file, downloads it,
    extracts it, and loads it into a Pandas DataFrame.
    """
    os.makedirs(download_dir, exist_ok=True)
    existing_maf = _find_existing_maf(download_dir)
    if existing_maf is not None and not force_download:
        print(f"Using existing MAF file: {existing_maf.name}")
        return pd.read_csv(existing_maf, sep='\t', comment='#', low_memory=False)
    
    # 1. Build the GDC API Query
    print("1. Querying NCI GDC API for the Aggregated TCGA-THCA MAF...")
    endpoint = "https://api.gdc.cancer.gov/files"
    
    # These filters target the Masked Somatic Mutations
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-THCA"]}},
            {"op": "in", "content": {"field": "data_category", "value": ["Simple Nucleotide Variation"]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Masked Somatic Mutation"]}},
            {"op": "in", "content": {"field": "data_format", "value": ["MAF"]}}
        ]
    }
    
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,analysis.workflow_type,file_size",
        "format": "JSON",
        "sort": "file_size:desc", # <--- THE FIX: The cohort files are the largest ones
        "size": "10" 
    }
    
    response = requests.get(endpoint, params=params)
    response.raise_for_status()
    hits = response.json()["data"]["hits"]
    
    # 2. Find the correct workflow (Ensemble or MuTect2)
    target_file = next(f for f in hits if "ensemble" in f.get("analysis", {}).get("workflow_type", "").lower() 
                       or "mutect" in f.get("analysis", {}).get("workflow_type", "").lower())
    
    file_id = target_file["file_id"]
    file_name = target_file["file_name"]
    file_size_mb = target_file.get("file_size", 0) / (1024 * 1024)
    
    gz_path = os.path.join(download_dir, file_name)
    maf_path = gz_path.replace(".gz", "")
    
    print(f"-> Found Cohort File: {file_name}")
    print(f"-> File Size: {file_size_mb:.2f} MB")
    
    if file_size_mb < 1.0:
        print("WARNING: This file is too small to be the full cohort. API structure may have changed.")
        
    # 3. Download the file
    print(f"\n2. Downloading (this may take a minute depending on connection)...")
    data_endpoint = f"https://api.gdc.cancer.gov/data/{file_id}"
    with requests.get(data_endpoint, stream=True) as r:
        r.raise_for_status()
        with open(gz_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
            
    # 4. Extract the .gz file
    print("3. Extracting MAF file...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(maf_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            
    # Clean up the compressed file to save space
    os.remove(gz_path)
    
    # 5. Load into Pandas
    print("4. Loading into Pandas DataFrame...")
    # MAF files have metadata headers starting with # that we need to skip
    df = pd.read_csv(maf_path, sep='\t', comment='#', low_memory=False)
    
    print(f"\nSuccess! Loaded {len(df)} total mutations across {df['Tumor_Sample_Barcode'].nunique()} patients.")
    return df


def ensure_tcga_thca_maf(download_dir: str | os.PathLike[str] = "./data") -> Path:
    existing_maf = _find_existing_maf(download_dir)
    if existing_maf is not None:
        return existing_maf
    df = download_tcga_thca_maf(download_dir=download_dir, force_download=False)
    if df.empty:
        raise RuntimeError("Downloaded MAF file is empty.")
    downloaded_maf = _find_existing_maf(download_dir)
    if downloaded_maf is None:
        raise FileNotFoundError("Expected a downloaded MAF file, but none was found on disk.")
    return downloaded_maf

if __name__ == "__main__":
    # Execute the pipeline
    maf_df = download_tcga_thca_maf()
    
    # Display the top 10 most frequently mutated genes in TCGA-THCA (BRAF will likely be #1)
    print("\nTop 10 Mutated Genes in TCGA-THCA:")
    print(maf_df['Hugo_Symbol'].value_counts().head(10))
