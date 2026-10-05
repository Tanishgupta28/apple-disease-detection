"""Download the released selected checkpoint and verify its saved checksum."""
import hashlib
import json
import sys
from pathlib import Path
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils import ROOT

def main():
    metadata=json.loads((ROOT/'models/selected_model.json').read_text())
    target=ROOT/metadata['checkpoint'];target.parent.mkdir(parents=True,exist_ok=True)
    expected=metadata['checkpoint_sha256']
    if not target.exists():
        url=f'https://github.com/Tanishgupta28/apple-disease-detection/releases/download/v1.0.0/{target.name}'
        partial=target.with_suffix('.pth.part')
        with requests.get(url,stream=True,timeout=(30,90)) as response:
            response.raise_for_status()
            with partial.open('wb') as f:
                for chunk in response.iter_content(1024*1024):
                    if chunk:f.write(chunk)
        digest=hashlib.sha256(partial.read_bytes()).hexdigest()
        if digest!=expected:
            partial.unlink();raise RuntimeError('Downloaded checkpoint checksum does not match the recorded benchmark')
        partial.replace(target)
    if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:raise RuntimeError('Existing checkpoint checksum mismatch')
    print('Verified selected checkpoint:',target)

if __name__=='__main__':main()
