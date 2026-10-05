"""GPU training with validation checkpoints and two-stage transfer learning."""
import argparse
import hashlib
import time
from pathlib import Path
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from .data import loaders, class_weights
from .models import build_model, configure_phase, training_mode
from .utils import ROOT, load_config, save_json, seed_everything

def run_epoch(model, loader, criterion, device, optimizer=None, architecture='custom_cnn'):
    """Measure a full epoch; gradient updates occur only for the training loader."""
    if optimizer is None:model.eval()
    else:training_mode(model, architecture)
    total_loss=0.;labels=[];predictions=[]
    with torch.set_grad_enabled(optimizer is not None):
        for images, target in loader:
            images=images.to(device, non_blocking=True);target=target.to(device, non_blocking=True)
            if optimizer is not None:optimizer.zero_grad(set_to_none=True)
            logits=model(images)
            loss=criterion(logits,target)
            if optimizer is not None:
                loss.backward();optimizer.step()
            total_loss+=loss.item()*len(target)
            labels.extend(target.detach().cpu().tolist())
            predictions.extend(logits.argmax(1).detach().cpu().tolist())
    return {'loss':total_loss/len(labels),'accuracy':float(accuracy_score(labels,predictions)),
            'macro_f1':float(f1_score(labels,predictions,labels=list(range(6)),average='macro',zero_division=0))}

def train_phase(model,name,phase,config,loaders_by_split,device,weights):
    """Save best validation macro-F1 checkpoint; use loss to break exact ties."""
    configure_phase(model,name,phase=='finetuned')
    epochs=config['cnn_epochs'] if name=='custom_cnn' else config['finetune_epochs' if phase=='finetuned' else 'head_epochs']
    lr=config['finetune_lr'] if phase=='finetuned' else config['head_lr']
    optimizer=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=lr)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=config['lr_patience'])
    training_loss=nn.CrossEntropyLoss(weight=weights.to(device))
    validation_loss=nn.CrossEntropyLoss()  # Uniform loss for comparable validation.
    experiment=f'{name}_{phase}' if name!='custom_cnn' else name
    checkpoint=ROOT/'models'/f'{experiment}.pth'
    history=[];best_score=-1.;best_loss=float('inf');stale=0;best_epoch=0;early_best=-1.
    start=time.perf_counter()
    print(f'EXPERIMENT {experiment}, device={device}, trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}',flush=True)
    for epoch in range(1,epochs+1):
        train=run_epoch(model,loaders_by_split['train'],training_loss,device,optimizer,name)
        validation=run_epoch(model,loaders_by_split['validation'],validation_loss,device)
        row={'epoch':epoch,'train':train,'validation':validation,'learning_rate':optimizer.param_groups[0]['lr']}
        history.append(row)
        print(f'{experiment} {epoch}/{epochs}: train acc={train["accuracy"]:.4f}; val acc={validation["accuracy"]:.4f}; val macro F1={validation["macro_f1"]:.4f}; val loss={validation["loss"]:.4f}',flush=True)
        score=validation['macro_f1']
        if score>best_score or (score==best_score and validation['loss']<best_loss):
            best_score=score;best_loss=validation['loss'];best_epoch=epoch
            metadata={'architecture':name,'experiment':experiment,'classes':config['classes'],
                'image_size':config['image_size'],'normalization':{'mean':[.485,.456,.406],'std':[.229,.224,.225]},
                'epoch':epoch,'validation':validation,'seed':config['seed'],
                'split_manifest_sha256':hashlib.sha256((ROOT/config['processed_dir']/'manifest.csv').read_bytes()).hexdigest(),
                'device':str(device),'gpu':torch.cuda.get_device_name() if device.type=='cuda' else None,
                'pretrained_weights':None if name=='custom_cnn' else ('IMAGENET1K_V1' if name=='mobilenetv2' else 'IMAGENET1K_V2')}
            torch.save({'state_dict':model.state_dict(),'metadata':metadata},checkpoint)
        if score>early_best+config['min_delta']:early_best=score;stale=0
        else:stale+=1
        scheduler.step(validation['loss'])
        save_json(ROOT/'results'/f'{experiment}_history.json',history)
        if stale>=config['patience']:
            print('Early stopping on validation macro F1',flush=True);break
    best=torch.load(checkpoint,map_location=device,weights_only=True);model.load_state_dict(best['state_dict'])
    summary={**best['metadata'],'checkpoint':str(checkpoint.relative_to(ROOT)),
        'best_epoch_training':history[best_epoch-1]['train'],'epochs_run':len(history),
        'parameter_count':sum(p.numel() for p in model.parameters()),
        'checkpoint_size_bytes':checkpoint.stat().st_size,'training_seconds':time.perf_counter()-start,
        'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    save_json(ROOT/'results'/f'{experiment}_summary.json',summary)
    plot_history(experiment,history)
    return summary

def plot_history(experiment,history):
    """Label full-epoch training and deterministic validation curves separately."""
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,key in zip(axes,['accuracy','loss']):
        for split in ['train','validation']:
            ax.plot([r['epoch'] for r in history],[r[split][key] for r in history],label=split)
        ax.set_xlabel('Epoch');ax.set_ylabel(key.title());ax.legend()
    fig.suptitle(experiment);fig.tight_layout()
    fig.savefig(ROOT/'results/figures'/f'{experiment}_history.png',dpi=140);plt.close(fig)

def refresh_comparison():
    """Persist real validation results and select a winner without opening test data."""
    import json
    names=['custom_cnn','mobilenetv2_frozen','mobilenetv2_finetuned','resnet50_frozen','resnet50_finetuned']
    summaries=[json.loads((ROOT/'results'/f'{n}_summary.json').read_text()) for n in names if (ROOT/'results'/f'{n}_summary.json').exists()]
    rows=[{'experiment':s['experiment'],'train_accuracy_at_best_epoch':s['best_epoch_training']['accuracy'],
        'validation_accuracy':s['validation']['accuracy'],'validation_macro_f1':s['validation']['macro_f1'],
        'validation_loss':s['validation']['loss'],'best_epoch':s['epoch'],'epochs_run':s['epochs_run'],
        'parameters':s['parameter_count'],'model_size_mb':s['checkpoint_size_bytes']/1e6} for s in summaries]
    pd.DataFrame(rows).to_csv(ROOT/'results/model_comparison.csv',index=False)
    if len(summaries)==5:
        winner=max(summaries,key=lambda s:(s['validation']['macro_f1'],-s['validation']['loss']))
        save_json(ROOT/'models/selected_model.json',winner)
        save_json(ROOT/'results/selection.json',{'selection_rule':'Maximum validation macro F1; ties by lower validation cross entropy.',
            'selected_experiment':winner['experiment'],'validation':winner['validation'],
            'test_accessed_during_training':False,'candidates':names,
            'checkpoint_sha256':winner['checkpoint_sha256']})
        print('SELECTED BY VALIDATION:',winner['experiment'],flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['all','custom_cnn','mobilenetv2','resnet50'],default='all')
    parser.add_argument('--config');parser.add_argument('--resume',action='store_true');args=parser.parse_args();config=load_config(args.config)
    if (ROOT/'results/metrics.json').exists():raise RuntimeError('This benchmark has already been tested. Preserve its results; use a deliberately separate benchmark for new training.')
    if config['require_cuda'] and not torch.cuda.is_available():raise RuntimeError('CUDA required: no visible GPU. Fix environment before training.')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('GPU:',torch.cuda.get_device_name() if device.type=='cuda' else 'CPU',flush=True)
    print('CUDA:',torch.version.cuda,'PyTorch:',torch.__version__,flush=True)
    torch.set_num_threads(8)
    summaries=[]
    for name in ['custom_cnn','mobilenetv2','resnet50'] if args.model=='all' else [args.model]:
        if args.resume and name=='custom_cnn' and (ROOT/'results/custom_cnn_summary.json').exists():
            print('Reusing completed custom CNN baseline.',flush=True);refresh_comparison();continue
        seed_everything(config['seed']);data=loaders(config);weights=class_weights(config)
        model=build_model(name).to(device)
        summaries.append(train_phase(model,name,'baseline' if name=='custom_cnn' else 'frozen',config,data,device,weights))
        refresh_comparison()
        if name!='custom_cnn':
            summaries.append(train_phase(model,name,'finetuned',config,data,device,weights));refresh_comparison()
        del model,data
        if device.type=='cuda':torch.cuda.empty_cache()
    print('Training completed.',flush=True)

if __name__=='__main__':main()
