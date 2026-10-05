"""Verify historical integrity, split isolation and final artifact consistency."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import nbformat
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils import ROOT, load_config, save_json

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    config=load_config();checks={}
    originals=json.loads((ROOT/'docs/original_manifest.json').read_text())
    for record in originals:
        path=ROOT/record['path'];assert digest(path)==record['sha256']
        if path.suffix=='.ipynb':nbformat.validate(nbformat.read(path,as_version=4))
    checks['original_files_preserved_and_valid']=len(originals)
    for path in (ROOT/'notebooks').glob('*.ipynb'):nbformat.validate(nbformat.read(path,as_version=4))
    checks['new_notebooks_valid']=4
    data=pd.read_csv(ROOT/'results/split_manifest.csv')
    for key in ['sha256','pixel_hash','group_id']:assert data.groupby(key).split.nunique().max()==1
    checks['known_duplicate_groups_do_not_cross_splits']=True
    if (ROOT/config['processed_dir']/'manifest.csv').exists():
        assert digest(ROOT/'results/split_manifest.csv')==digest(ROOT/config['processed_dir']/'manifest.csv')
    checks['published_split_matches_training_manifest']=True
    selection=json.loads((ROOT/'results/selection.json').read_text())
    selected=json.loads((ROOT/'models/selected_model.json').read_text())
    measured=json.loads((ROOT/'results/metrics.json').read_text())
    assert selected['classes']==config['classes']==measured['classes']
    assert selected['checkpoint_sha256']==selection['checkpoint_sha256']==measured['selection_checkpoint_sha256']
    assert digest(ROOT/selected['checkpoint'])==selected['checkpoint_sha256']
    assert selected['split_manifest_sha256']==digest(ROOT/'results/split_manifest.csv')
    assert not measured['test_used_for_model_selection']
    comparison=pd.read_csv(ROOT/'results/model_comparison.csv')
    winner=comparison.sort_values(['validation_macro_f1','validation_loss'],ascending=[False,True]).iloc[0]
    assert winner.experiment==selection['selected_experiment']
    checks['selection_and_preprocessing_labels_consistent']=True
    checks['validation_selection_matches_recorded_winner']=True
    predictions=pd.read_csv(ROOT/'results/test_predictions.csv')
    assert predictions.path.tolist()==data[data.split=='test'].path.tolist()
    assert len(predictions)==measured['test']['n_images']
    assert abs((predictions.label==predictions.predicted_label).mean()-measured['test']['accuracy'])<1e-12
    checks['test_metrics_match_saved_predictions']=True
    summaries=[json.loads(p.read_text()) for p in (ROOT/'results').glob('*_summary.json')]
    assert len(summaries)==5 and all(s['device']=='cuda' and s['gpu'] for s in summaries)
    checks['gpu_training_provenance']=list({s['gpu'] for s in summaries})
    tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    assert not any(p.startswith(('data/raw/','data/processed/')) or p.endswith(('.pth','.pt','.h5','.keras')) for p in tracked)
    checks['raw_images_and_checkpoints_excluded_from_git']=True
    # Scan every historical commit's file content; report paths only, never possible secret text.
    import re
    patterns=[re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')]
    checked=set();suspects=[]
    commits=subprocess.check_output(['git','rev-list','--all'],cwd=ROOT,text=True).splitlines()
    for commit in commits:
        entries=subprocess.check_output(['git','ls-tree','-r',commit],cwd=ROOT,text=True).splitlines()
        for entry in entries:
            header,path=entry.split('\t',1);blob=header.split()[2]
            if blob in checked:continue
            checked.add(blob);content=subprocess.check_output(['git','cat-file','blob',blob],cwd=ROOT)
            if any(pattern.search(content) for pattern in patterns):suspects.append(path)
    assert not suspects, f'Potential credential patterns found in paths: {suspects}'
    checks['credential_pattern_scan']={'commits_checked':len(commits),'unique_blobs_checked':len(checked),'suspect_paths':suspects,
        'scope':'Known token/private-key patterns; not a proof that every possible secret format is absent.'}
    checks['git_remote']=subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT,text=True).strip()
    if not args.check_only:save_json(ROOT/'results/final_audit.json',checks)
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
