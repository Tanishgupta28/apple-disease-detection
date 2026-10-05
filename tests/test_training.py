"""Check full-epoch accounting and safe checkpoint metric serialization."""
import torch
from torch.utils.data import DataLoader,TensorDataset
from src.train import run_epoch

def test_validation_metrics_are_plain_python_scalars_and_count_all_batches(tmp_path):
    class PredictFromPixel(torch.nn.Module):
        def forward(self,x):return x[:, :, 0, 0]
    x=torch.zeros(5,6,1,1);labels=torch.tensor([0,1,2,3,4])
    x[range(5),labels,0,0]=5
    result=run_epoch(PredictFromPixel(),DataLoader(TensorDataset(x,labels),batch_size=2),torch.nn.CrossEntropyLoss(),torch.device('cpu'))
    assert result['accuracy']==1.0
    assert all(type(value) is float for value in result.values())
    path=tmp_path/'test.pth';torch.save({'metadata':result,'state_dict':{}},path)
    assert torch.load(path,weights_only=True)['metadata']==result
