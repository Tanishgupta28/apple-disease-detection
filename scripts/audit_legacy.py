"""Audit preserved notebooks without executing or rewriting them."""
import hashlib
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
lines = ['# Historical NIT internship experiments', '',
'These notebooks are historical work, not the upgraded six-class leaf benchmark. No old notebook was rerun. Evidence below is recovered from source cells and stored outputs; cell numbers are zero-based.', '',
'## Recovery and dataset', '',
'All nine publicly listed Drive files were downloaded, kept byte-for-byte under `original_work/Apple disease detection codes/`, and validated with nbformat. SHA-256 hashes and sizes are in `original_manifest.json`. No dataset, standalone checkpoint, or additional project file was present in the public folder listing.', '',
'Every notebook has stored generator outputs confirming **307 training, 75 validation, and 120 test images** (502 total), four classes. Source paths are `/kaggle/input/appledisease/Train` and `/kaggle/input/appledisease/Test`. Stored file listings identify Blotch_Apple, Rot_Apple, Scab_Apple, and Normal_Apple. There is no recoverable source URL/license or original image archive. The labels and filenames do not establish that the dataset consists exclusively of leaves; this cannot be verified without its images.', '',
'## Comparison', '',
'| Notebook | Last logged train accuracy by phase* | Maximum logged validation accuracy by phase | Genuine stored test accuracy |',
'|---|---|---|---|']
sections=[]
for p in sorted((root/'original_work').rglob('*.ipynb')):
 n=json.loads(p.read_text());phases=[];tests=[];reported=[];evidence=[]
 for i,c in enumerate(n['cells']):
  s=''.join(c.get('source',[]))
  for o in c.get('outputs',[]):
   t=''.join(o.get('text',o.get('data',{}).get('text/plain','')))
   t=re.sub(r'\x1b\[[0-9;]*m','',t)
   ms=re.findall(r'accuracy: ([0-9.]+) - loss: ([0-9.]+) - val_accuracy: ([0-9.]+) - val_loss: ([0-9.]+)',t)
   if ms: phases.append({'cell':i,'logged_epochs':len(ms),'last_train':ms[-1][0],'max_val':max(float(m[2]) for m in ms),'last_val':ms[-1][2]})
   if 'Test Accuracy:' in t or 'ENSEMBLE ACCURACY:' in t:tests.append(t.strip())
   if 'Best Training Accuracy:' in t or 'Best Validation Accuracy:' in t:reported.append((i,t.strip()))
  relevant=['img_size =','ImageDataGenerator(','model =','base_model =','trainable','compile(','model.fit','model.save','evaluate(','EarlyStopping(','ReduceLROnPlateau(','fine_tune_at','compute_class_weight','ensemble_preds =']
  if c['cell_type']=='code' and any(k in s for k in relevant):evidence.append((i,s))
 lines.append('| '+p.name+' | '+' / '.join(x['last_train'] for x in phases)+' | '+' / '.join(f"{x['max_val']:.4f}" for x in phases)+' | '+('; '.join(tests) or 'Not stored')+' |')
 sections += ['', '## '+p.name,'',f'File: `{p.relative_to(root)}`. No embedded PNG figures. No stored precision/recall/F1 report. Imported plotting libraries alone do not establish plotted results. No written conclusions were stored.', '',
 '### Training evidence', '',json.dumps(phases,indent=2), '', '### Printed summaries', '']
 sections += [f'- Cell {i}: {t.replace(chr(10), "; ")}' for i,t in reported] or ['Not stored.']
 sections += ['', '### Architecture, preprocessing, weights, fine-tuning, optimizer and callback source', '']
 for i,s in evidence:sections += [f'Cell {i}:','```python',s,'```','']
sections += ['## Audit interpretation','',
'*Last logged training accuracies in the comparison are progress-bar measurements, not exact full-epoch history values. Printed best-training summaries above are separate evidence. Maxima over validation logs need not correspond to the checkpoint restored by minimum validation loss.', '',
'- All transfer models load ImageNet weights and use global average pooling plus a dense softmax head. DenseNet121 and InceptionV3 unfreeze the full backbone; other notebook source shows partial unfreezing.',
'- The baseline and ensemble use the same augmented generator for validation; validation predictions therefore vary under augmentation. Most separate transfer notebooks provide a separate unaugmented validation generator.',
'- MobileNetV2, DenseNet121, InceptionV3 and Xception use only rescaling to [0,1], rather than their Keras ImageNet-specific preprocessing. ConvNeXt also rescales despite its built-in preprocessing. ResNet50 and EfficientNetB3 explicitly use their application preprocess_input.',
'- No explicit deterministic split seed, duplicate audit, untouched-test model selection policy, or reusable inference metadata was recovered.',
'- The CNN has 8,530,596 parameters, dominated by Flatten → Dense(256), despite only 307 training images. Its test evaluation source cell has no output, so test accuracy is unknown.',
'- MobileNetV2 and ResNet50 fine-tuning validation maxima are below their frozen-phase maxima. Fine-tuning is not automatically an improvement.',
'- Xception printed 0.8933 for fine-tuning validation, then 0.8666666746 from the frozen-phase history in a later cell. This is a history-variable mismatch, not test evidence.',
'- ConvNeXtBase is a convolutional architecture; a filename mentioning transformers does not make it a transformer.',
'- Ensemble averages ResNet50, DenseNet121 and EfficientNetB0 probabilities. It genuinely reports 73.33% on the 120-image test set. It saves three .h5 filenames in source/output, but none were present in Drive.',
'- Apart from ConvNeXt (57.50%) and the ensemble (73.33%), genuine stored test accuracy is unavailable. Validation values such as EfficientNetB3 96% are not test accuracy.',
'- The old ensemble logs include a failed CUDA initialization and dependency conflicts. These are historical logs; current GPU operation is independently recorded in environment.json.',
'- The new benchmark changes both classes and dataset. Its numbers must not be presented as a controlled improvement over the old four-class results.']
(root/'docs/legacy_experiments.md').write_text('\n'.join(lines+sections)+'\n')
print('Audited',len(list((root/'original_work').rglob('*.ipynb'))),'notebooks; wrote docs/legacy_experiments.md')
