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

def test_training_phase_saves_reloadable_checkpoint_in_isolated_run(tmp_path,monkeypatch):
    import src.train as training
    from src.models import build_model
    monkeypatch.setattr(training,'ROOT',tmp_path)
    manifest=tmp_path/'data/processed/manifest.csv';manifest.parent.mkdir(parents=True);manifest.write_text('fixture manifest\n')
    run_root=tmp_path/'runs/smoke'
    (run_root/'models').mkdir(parents=True);(run_root/'results/figures').mkdir(parents=True)
    config={'cnn_epochs':1,'head_lr':.001,'lr_patience':2,'classes':['a','b','c','d','e','f'],
            'image_size':32,'seed':42,'processed_dir':'data/processed','min_delta':.0005,'patience':5}
    torch.manual_seed(42)
    images=torch.randn(6,3,32,32);labels=torch.arange(6)
    loader=DataLoader(TensorDataset(images,labels),batch_size=3)
    model=build_model('custom_cnn',pretrained=False)
    before=model.classifier[-1].weight.detach().clone()
    summary=training.train_phase(model,'custom_cnn','baseline',config,
        {'train':loader,'validation':loader},torch.device('cpu'),torch.ones(6),run_root)
    path=tmp_path/summary['checkpoint']
    assert path.is_relative_to(run_root)
    saved=torch.load(path,weights_only=True)
    assert saved['metadata']['classes']==config['classes']
    assert not torch.equal(before,model.classifier[-1].weight)
    assert (run_root/'results/custom_cnn_summary.json').exists()
