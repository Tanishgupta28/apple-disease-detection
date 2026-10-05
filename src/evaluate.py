"""Evaluate the validation-selected model once on the held-out test split."""
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from torch.utils.data import DataLoader
from .data import loaders, LeafDataset, preprocess
from .gradcam import gradcam, overlay
from .predict import load_predictor
from .utils import ROOT, load_config, save_json

def metrics(y,p,classes):
    """Summarize measured accuracy and balanced/classwise classification metrics."""
    macro=precision_recall_fscore_support(y,p,labels=list(range(6)),average='macro',zero_division=0)
    weighted=precision_recall_fscore_support(y,p,labels=list(range(6)),average='weighted',zero_division=0)
    return {'n_images':len(y),'accuracy':accuracy_score(y,p),'macro_precision':macro[0],
        'macro_recall':macro[1],'macro_f1':macro[2],'weighted_f1':weighted[2],
        'classwise':classification_report(y,p,labels=list(range(6)),target_names=classes,output_dict=True,zero_division=0),
        'confusion_matrix':confusion_matrix(y,p,labels=list(range(6))).tolist()}

def collect(model,loader,device):
    """Collect ordered probabilities without gradient updates."""
    targets=[];probabilities=[]
    model.eval()
    with torch.inference_mode():
        for images,target in loader:
            probabilities.extend(model(images.to(device,non_blocking=True)).softmax(1).cpu().numpy())
            targets.extend(target.tolist())
    return np.asarray(targets),np.asarray(probabilities)

def benchmark(model,metadata,device,image):
    """Warm up then time synchronized batch-one GPU forward and end-to-end inference."""
    tensor=preprocess(image,metadata['image_size']).unsqueeze(0).to(device)
    sync=lambda:torch.cuda.synchronize() if device.type=='cuda' else None
    with torch.inference_mode():
        for _ in range(10):model(tensor)
        sync();forward=[];end_to_end=[]
        for _ in range(50):
            sync();start=time.perf_counter();model(tensor).softmax(1).cpu();sync()
            forward.append((time.perf_counter()-start)*1000)
        for _ in range(20):
            sync();start=time.perf_counter()
            x=preprocess(image,metadata['image_size']).unsqueeze(0).to(device)
            model(x).softmax(1).cpu();sync()
            end_to_end.append((time.perf_counter()-start)*1000)
    return {'device':str(device),'hardware':torch.cuda.get_device_name() if device.type=='cuda' else 'CPU',
        'batch_size':1,'forward_samples':50,'forward_median_ms':float(np.median(forward)),
        'forward_p95_ms':float(np.percentile(forward,95)),
        'preprocess_and_forward_median_ms':float(np.median(end_to_end)),
        'excludes':'Model loading, file decoding, upload, and Grad-CAM.'}

def figures(report,rows,probabilities,model,metadata,device,run_root=ROOT):
    """Plot confusion/F1, selected errors, and representative Grad-CAM examples."""
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    classes=metadata['classes'];cm=np.asarray(report['test']['confusion_matrix'])
    fig,ax=plt.subplots(figsize=(10,8));im=ax.imshow(cm,cmap='Blues');fig.colorbar(im,ax=ax)
    ax.set_xticks(range(6),classes,rotation=35,ha='right');ax.set_yticks(range(6),classes)
    ax.set_xlabel('Predicted class');ax.set_ylabel('True class');ax.set_title('Untouched test set — raw counts')
    for i in range(6):
        for j in range(6):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    fig.tight_layout();fig.savefig(run_root/'results/figures/confusion_matrix.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5));ax.bar(classes,[report['test']['classwise'][c]['f1-score'] for c in classes])
    ax.set_ylim(0,1);ax.set_ylabel('Test F1');ax.tick_params(axis='x',rotation=20)
    fig.tight_layout();fig.savefig(run_root/'results/figures/classwise_f1.png',dpi=150);plt.close(fig)
    y=rows.label.to_numpy();p=probabilities.argmax(1);confidence=probabilities.max(1)
    correct=np.where(y==p)[0];wrong=np.where(y!=p)[0];low=np.where(confidence<.6)[0]
    categories={'correct_high_confidence':correct[np.argsort(-confidence[correct])][:6],
        'wrong_high_confidence':wrong[np.argsort(-confidence[wrong])][:6],
        'low_confidence':low[np.argsort(confidence[low])][:6]}
    examples={}
    for category,indices in categories.items():
        if not len(indices):examples[category]=[];continue
        fig,axes=plt.subplots(2,3,figsize=(13,8));examples[category]=[]
        for ax in axes.flat:ax.axis('off')
        for ax,index in zip(axes.flat,indices):
            row=rows.iloc[index]
            with Image.open(ROOT/row.path) as im:ax.imshow(ImageOps.exif_transpose(im))
            ax.set_title(f'True: {classes[y[index]]}\nPredicted: {classes[p[index]]} ({confidence[index]:.1%})',fontsize=10)
            examples[category].append({'path':row.path,'true':classes[y[index]],'predicted':classes[p[index]],'confidence':float(confidence[index])})
        fig.suptitle(category.replace('_',' '));fig.tight_layout()
        fig.savefig(run_root/'results/figures'/f'{category}.png',dpi=130);plt.close(fig)
    save_json(run_root/'results/error_examples.json',examples)
    representatives=[]
    for label in range(6):
        candidates=np.where(y==label)[0]
        if len(candidates):representatives.append(int(candidates[0]))
    fig,axes=plt.subplots(len(representatives),3,figsize=(12,4*len(representatives)))
    for row_index,index in enumerate(representatives):
        row=rows.iloc[index]
        with Image.open(ROOT/row.path) as im:image=ImageOps.exif_transpose(im).convert('RGB')
        tensor=preprocess(image,metadata['image_size']).unsqueeze(0).to(device)
        heatmap,predicted=gradcam(model,metadata['architecture'],tensor)
        for ax in axes[row_index]:ax.axis('off')
        axes[row_index,0].imshow(image);axes[row_index,0].set_title(f'True: {classes[y[index]]}')
        axes[row_index,1].imshow(heatmap,cmap='jet');axes[row_index,1].set_title('Grad-CAM (predicted class)')
        axes[row_index,2].imshow(overlay(image,heatmap));axes[row_index,2].set_title(f'{classes[predicted]} ({confidence[index]:.1%})')
    fig.tight_layout();fig.savefig(run_root/'results/figures/gradcam_examples.png',dpi=100);plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config');parser.add_argument('--run-dir',default='.');args=parser.parse_args();config=load_config(args.config)
    run_root=(ROOT/args.run_dir).resolve()
    if not run_root.is_relative_to(ROOT):raise ValueError('Run directory must stay within this project root')
    selection=json.loads((run_root/'results/selection.json').read_text())
    output=run_root/'results/metrics.json'
    if output.exists():
        print('Test results already exist. Reusing the frozen evaluation; no retraining or repeated test selection.');return
    selected=json.loads((run_root/'models/selected_model.json').read_text())
    model,metadata,device=load_predictor(ROOT/selected['checkpoint']);torch.set_num_threads(8)
    checkpoint=ROOT/selected['checkpoint']
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=selection['checkpoint_sha256']:
        raise ValueError('Selected checkpoint changed after validation selection')
    manifest=ROOT/config['processed_dir']/'manifest.csv'
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=metadata['split_manifest_sha256']:
        raise ValueError('Split manifest changed after training')
    data=loaders(config,splits=('validation','test'));rows=pd.read_csv(manifest)
    train_dataset=LeafDataset(rows[rows.split=='train'],config['image_size'],training=False)
    data['train']=DataLoader(train_dataset,batch_size=config['batch_size'],num_workers=config['num_workers'],pin_memory=device.type=='cuda')
    report={'selected_experiment':selection['selected_experiment'],'classes':metadata['classes'],
        'selection_rule':selection['selection_rule'],'selection_checkpoint_sha256':selection['checkpoint_sha256'],
        'test_used_for_model_selection':False,'train_metrics_use_augmentation':False}
    test_probabilities=None
    for split in ('train','validation','test'):
        y,probabilities=collect(model,data[split],device)
        report[split]=metrics(y,probabilities.argmax(1),metadata['classes'])
        print(split,report[split]['accuracy'],report[split]['macro_f1'],flush=True)
        if split=='test':test_probabilities=probabilities
    test_rows=rows[rows.split=='test'].reset_index(drop=True)
    pred=test_probabilities.argmax(1);confidence=test_probabilities.max(1)
    predictions=test_rows[['path','label','group_id']].copy()
    predictions['predicted_label']=pred;predictions['confidence']=confidence
    for i,c in enumerate(metadata['classes']):predictions[f'probability_{c}']=test_probabilities[:,i]
    predictions.to_csv(run_root/'results/test_predictions.csv',index=False)
    cm=np.asarray(report['test']['confusion_matrix']);confusions=[]
    for i in range(6):
        for j in range(i+1,6):
            confusions.append({'classes':[metadata['classes'][i],metadata['classes'][j]],
                'total_bidirectional_errors':int(cm[i,j]+cm[j,i]),
                'first_to_second':int(cm[i,j]),'second_to_first':int(cm[j,i])})
    report['major_confusions']=sorted(confusions,key=lambda r:r['total_bidirectional_errors'],reverse=True)[:5]
    with Image.open(ROOT/test_rows.iloc[0].path) as im:report['inference']=benchmark(model,metadata,device,im.convert('RGB'))
    report['parameter_count']=selected['parameter_count'];report['checkpoint_size_bytes']=checkpoint.stat().st_size
    figures(report,test_rows,test_probabilities,model,metadata,device,run_root)
    save_json(output,report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
