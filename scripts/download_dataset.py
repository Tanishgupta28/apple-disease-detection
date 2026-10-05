"""Download and safely extract the public official AppleLeaf9 archive."""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils import ROOT

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--url',default='https://github.com/JasonYangCode/AppleLeaf9/archive/refs/heads/main.zip')
    args=parser.parse_args();directory=ROOT/'data/raw';directory.mkdir(parents=True,exist_ok=True)
    archive=directory/'AppleLeaf9-main.zip';partial=archive.with_suffix('.zip.part')
    if not archive.exists():
        with requests.get(args.url,stream=True,timeout=(30,90)) as response:
            response.raise_for_status()
            with partial.open('wb') as output:
                for chunk in response.iter_content(1024*1024):
                    if chunk:output.write(chunk)
        partial.replace(archive)
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    evidence=ROOT/'results/dataset_audit.json'
    if evidence.exists():
        expected=json.loads(evidence.read_text())['archive_sha256']
        if digest!=expected:raise RuntimeError('Source archive changed from the measured benchmark. Do not reuse published split/results with different data.')
    target=directory/'AppleLeaf9-main'
    if not target.exists():
        with zipfile.ZipFile(archive) as source:
            for info in source.infolist():
                if not (directory/info.filename).resolve().is_relative_to(directory.resolve()):raise ValueError('Unsafe archive path')
            source.extractall(directory)
    print('Official dataset:',target,'archive SHA-256:',digest)

if __name__=='__main__':main()
