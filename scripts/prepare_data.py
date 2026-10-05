"""Audit AppleLeaf9, remove exact duplicates, and group related images before splitting."""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.fft import dctn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils import ROOT, load_config, save_json

SOURCE_CLASSES = {'Scab':0, 'Rust':1, 'Frogeye leaf spot':2, 'Powdery mildew':3,
                  'Alternaria leaf spot':4, 'Health':5}

def phash(image):
    """64-bit low-frequency perceptual hash."""
    pixels = np.asarray(image.convert('L').resize((32,32)), dtype=float)
    low = dctn(pixels, norm='ortho')[:8,:8]
    bits = low > np.median(low.flatten()[1:])
    return int.from_bytes(np.packbits(bits.flatten()).tobytes(), 'big')

def inspect(item):
    path,label = item
    record = {'path':str(path.relative_to(ROOT)), 'label':label,
              'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            image=ImageOps.exif_transpose(im).convert('RGB');image.load()
            record.update(width=image.width,height=image.height)
            record['pixel_hash']=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
            small=image.resize((128,128))
            hashes=[]
            for angle in (0,90,180,270):
                rotated=small.rotate(angle)
                hashes += [phash(rotated),phash(ImageOps.mirror(rotated))]
            record['phashes']=hashes
    except Exception as e:
        record['error']=str(e)
    return record

class BKTree:
    """Index perceptual hashes by Hamming distance without quadratic storage."""
    def __init__(self):self.root=None
    def add(self,value,index):
        if self.root is None:self.root=[value,[index],{}];return
        node=self.root
        while True:
            distance=(value^node[0]).bit_count()
            if distance==0:node[1].append(index);return
            if distance not in node[2]:node[2][distance]=[value,[index],{}];return
            node=node[2][distance]
    def query(self,value,radius):
        if self.root is None:return []
        stack=[self.root];result=[]
        while stack:
            node=stack.pop();distance=(value^node[0]).bit_count()
            if distance<=radius:result.extend(node[1])
            stack.extend(child for edge,child in node[2].items() if distance-radius<=edge<=distance+radius)
        return result

def main():
    config=load_config();raw=ROOT/config['raw_dir'];processed=ROOT/config['processed_dir']
    if (processed/'manifest.csv').exists():
        raise RuntimeError('Existing split is immutable. Use a fresh configured processed directory for a new benchmark.')
    items=[(p,label) for folder,label in SOURCE_CLASSES.items()
           for p in sorted((raw/folder).rglob('*')) if p.suffix.lower() in {'.jpg','.jpeg','.png'}]
    with ThreadPoolExecutor(max_workers=12) as pool:records=list(pool.map(inspect,items))
    corrupt=[r for r in records if 'error' in r]
    valid=[r for r in records if 'error' not in r]
    # Exclude contradictory exact/pixel duplicates instead of choosing arbitrary labels.
    by_pixel=defaultdict(list)
    for r in valid:by_pixel[r['pixel_hash']].append(r)
    conflicts=[];dedup=[];removed=[]
    for group in by_pixel.values():
        if len({r['label'] for r in group})>1:
            conflicts.extend(group);continue
        dedup.append(group[0]);removed.extend(group[1:])
    parent=list(range(len(dedup)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    tree=BKTree();pairs=[]
    # Compare normal hash against all earlier rotations/reflections, radius 4/64.
    for i,r in enumerate(dedup):
        for j in tree.query(r['phashes'][0],4):
            union(i,j)
            if len(pairs)<200:pairs.append((i,j))
        for h in set(r['phashes']):tree.add(h,i)
        if (i+1)%2000==0:print('Grouped',i+1,'images',flush=True)
    groups=defaultdict(list)
    for i,r in enumerate(dedup):groups[find(i)].append(r)
    # Near duplicate label conflicts are quarantined conservatively.
    usable=[];near_conflicts=[]
    for group in groups.values():
        if len({r['label'] for r in group})>1:near_conflicts.extend(group)
        else:usable.append(group)
    rng=np.random.default_rng(config['seed']);final=[]
    for label in range(6):
        class_groups=[g for g in usable if g[0]['label']==label]
        rng.shuffle(class_groups)
        class_groups.sort(key=len,reverse=True)
        total=sum(map(len,class_groups));target=np.array([.7,.15,.15])*total
        counts=np.zeros(3,dtype=int)
        for g in class_groups:
            split_index=int(np.argmax((target-counts)/np.maximum(target,1)))
            counts[split_index]+=len(g)
            group_id=min(r['sha256'] for r in g)
            for r in g:final.append({**r,'group_id':group_id,'split':['train','validation','test'][split_index]})
    processed.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(final).drop(columns=['phashes'])
    df.to_csv(processed/'manifest.csv',index=False)
    # Symlinks expose a familiar folder layout without duplicating raw images.
    for r in final:
        directory=processed/r['split']/config['classes'][r['label']];directory.mkdir(parents=True,exist_ok=True)
        dst=directory/(r['sha256'][:12]+'_'+Path(r['path']).name)
        dst.symlink_to(ROOT/r['path'])
    assert df.groupby('group_id').split.nunique().max()==1
    assert df.groupby('sha256').split.nunique().max()==1
    assert df.groupby('pixel_hash').split.nunique().max()==1
    all_counts={folder:sum(p.suffix.lower() in {'.jpg','.jpeg','.png'} for p in (raw/folder).rglob('*'))
                for folder in ['Scab','Rust','Frogeye leaf spot','Powdery mildew','Alternaria leaf spot','Health','Brown spot','Grey spot','Mosaic']}
    report={'source':'https://github.com/JasonYangCode/AppleLeaf9','license':'CC BY 4.0',
        'citation':'Yang, Q.; Duan, S.; Wang, L. (2022). Efficient Identification of Apple Leaf Diseases in the Wild Using Convolutional Neural Networks. Agronomy 12(11), 2784. https://doi.org/10.3390/agronomy12112784',
        'archive_sha256':hashlib.sha256((ROOT/'data/raw/AppleLeaf9-main.zip').read_bytes()).hexdigest(),
        'source_class_counts':all_counts,'source_total':sum(all_counts.values()),'selected_before_audit':len(items),
        'classes':config['classes'],'excluded_classes':['Brown spot','Grey spot','Mosaic'],
        'corrupt_images':corrupt,'exact_or_decoded_duplicates_removed':len(removed),
        'exact_label_conflicts_removed':len(conflicts),'near_duplicate_label_conflicts_removed':len(near_conflicts),
        'images_used':len(df),'groups_used':len(usable),'perceptual_hamming_threshold':4,
        'grouping':'Connected components of pHash matches across rotations/reflections; no group spans splits.',
        'seed':config['seed'],'class_counts':{config['classes'][i]:int((df.label==i).sum()) for i in range(6)},
        'split_counts':df.split.value_counts().to_dict(),
        'split_class_counts':{s:{config['classes'][i]:int(((df.split==s)&(df.label==i)).sum()) for i in range(6)} for s in ['train','validation','test']},
        'dimension_summary':df[['width','height']].describe().to_dict(),
        'split_manifest_sha256':hashlib.sha256((processed/'manifest.csv').read_bytes()).hexdigest(),
        'limitations':'Perceptual grouping cannot establish orchard/leaf identities or detect every crop/photographic variant; source provides no acquisition group IDs. No absolute claim of absence of leakage is possible.'}
    save_json(ROOT/'results/dataset_audit.json',report)
    save_json(ROOT/'data/processed/audit_details.json',{'duplicates_removed':removed,'conflicts':conflicts+near_conflicts})
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(6,4,figsize=(12,16))
    for label in range(6):
        subset=df[(df.label==label)&(df.split=='train')].head(4)
        for axis,(_,row) in zip(axes[label],subset.iterrows()):
            with Image.open(ROOT/row.path) as im:axis.imshow(im)
            axis.set_title(config['classes'][label]);axis.axis('off')
    fig.tight_layout();fig.savefig(ROOT/'results/figures/training_samples.png',dpi=130);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5));ax.bar(report['class_counts'].keys(),report['class_counts'].values())
    ax.tick_params(axis='x',rotation=20);ax.set_ylabel('Images after audit');fig.tight_layout()
    fig.savefig(ROOT/'results/figures/class_distribution.png',dpi=140);plt.close(fig)
    # A sample of matched pairs supports manual review of the grouping method.
    if pairs:
        fig,axes=plt.subplots(8,2,figsize=(9,24))
        for row,(a,b) in enumerate(pairs[:8]):
            for axis,index in zip(axes[row],[a,b]):
                record=dedup[index]
                with Image.open(ROOT/record['path']) as im:axis.imshow(im)
                axis.set_title(config['classes'][record['label']]);axis.axis('off')
        fig.tight_layout();fig.savefig(ROOT/'results/figures/duplicate_review.png',dpi=90);plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
