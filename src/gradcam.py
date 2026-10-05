"""Gradient-weighted class activation maps for the predicted convolutional class."""
import numpy as np
import torch
from PIL import Image
from .models import gradcam_layer

def gradcam(model,architecture,tensor,class_index=None):
    """Return a normalized heatmap and class index; release hooks after each call."""
    activations=[];gradients=[]
    def capture(module,inputs,output):
        activations.append(output)
        output.register_hook(lambda gradient:gradients.append(gradient))
    handle=gradcam_layer(model,architecture).register_forward_hook(capture)
    try:
        model.eval();model.zero_grad(set_to_none=True)
        x=tensor.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            logits=model(x)
            index=int(logits.argmax(1).item()) if class_index is None else int(class_index)
            logits[0,index].backward()
            weights=gradients[0].mean(dim=(2,3),keepdim=True)
            heatmap=(weights*activations[0]).sum(1).relu()
            heatmap=torch.nn.functional.interpolate(heatmap.unsqueeze(1),size=x.shape[-2:],mode='bilinear',align_corners=False)[0,0]
            heatmap=heatmap-heatmap.min();heatmap=heatmap/heatmap.max().clamp_min(1e-8)
        return heatmap.detach().cpu().numpy(),index
    finally:
        handle.remove();model.zero_grad(set_to_none=True)

def overlay(image,heatmap,alpha=.4):
    """Blend a display heatmap with the corresponding original RGB image."""
    from matplotlib import colormaps
    base=image.convert('RGB')
    heat=Image.fromarray(np.uint8(heatmap*255)).resize(base.size,Image.Resampling.BILINEAR)
    colors=colormaps['jet'](np.asarray(heat)/255.)[:,:,:3]
    colored=Image.fromarray(np.uint8(colors*255))
    return Image.blend(base,colored,alpha)
