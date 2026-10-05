"""Inference with checkpoint label order and shared preprocessing."""
import argparse
import json
import torch
from PIL import Image
from .data import preprocess
from .models import build_model
from .utils import ROOT

def load_predictor(checkpoint=None,device=None):
    """Reconstruct a saved architecture without downloading ImageNet weights."""
    if checkpoint is None:
        selected=ROOT/'models/selected_model.json'
        if not selected.exists():raise FileNotFoundError('Train all three models first to create models/selected_model.json.')
        checkpoint=ROOT/json.loads(selected.read_text())['checkpoint']
    device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    saved=torch.load(checkpoint,map_location=device,weights_only=True)
    metadata=saved['metadata']
    if len(metadata['classes'])!=6 or len(set(metadata['classes']))!=6:raise ValueError('Invalid six-class checkpoint label mapping')
    model=build_model(metadata['architecture'],pretrained=False).to(device)
    model.load_state_dict(saved['state_dict']);model.eval()
    return model,metadata,device

def predict_image(image,model,metadata,device):
    """Return all six probabilities in checkpoint order and a sorted ranking."""
    tensor=preprocess(image,metadata['image_size']).unsqueeze(0).to(device)
    with torch.inference_mode():probabilities=model(tensor).softmax(1)[0].cpu().tolist()
    ranked=sorted(zip(metadata['classes'],probabilities),key=lambda p:p[1],reverse=True)
    return {'predicted_class':ranked[0][0],'confidence':ranked[0][1],
            'top_predictions':[{'class':label,'probability':p} for label,p in ranked],
            'probabilities':dict(zip(metadata['classes'],probabilities))}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('image');parser.add_argument('--checkpoint');args=parser.parse_args()
    model,metadata,device=load_predictor(args.checkpoint)
    with Image.open(args.image) as im:print(json.dumps(predict_image(im,model,metadata,device),indent=2))

if __name__=='__main__':main()
