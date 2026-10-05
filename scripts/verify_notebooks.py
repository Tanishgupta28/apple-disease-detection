"""Execute teaching notebooks safely with their training toggles disabled."""
import sys
import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils import ROOT, save_json

def main():
    report=[]
    for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
        notebook=nbformat.read(path,as_version=4)
        assert all('RUN_TRAINING = True' not in cell.source for cell in notebook.cells)
        start=time.perf_counter()
        NotebookClient(notebook,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
        errors=[output for cell in notebook.cells if cell.cell_type=='code' for output in cell.get('outputs',[]) if output.output_type=='error']
        assert not errors
        report.append({'notebook':str(path.relative_to(ROOT)),'executed_without_errors':True,'execution_seconds':time.perf_counter()-start,
                       'training_toggles_disabled':True})
        print(report[-1],flush=True)
    # Verify execution without duplicating figures/large outputs inside notebook files.
    save_json(ROOT/'results/notebook_verification.json',report)

if __name__=='__main__':main()
