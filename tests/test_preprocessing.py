"""Preprocessing contracts independent of the full dataset."""
import numpy as np
import torch
from PIL import Image
from src.data import preprocess, transform, MEAN, STD

def test_preprocessing_is_deterministic_rgb_and_same_as_validation():
    image=Image.fromarray(np.random.default_rng(42).integers(0,256,(80,110,3),dtype=np.uint8))
    a=preprocess(image,64);b=preprocess(image,64)
    assert a.shape==(3,64,64)
    assert a.dtype==torch.float32
    assert torch.equal(a,b)
    assert torch.equal(a,transform(64,training=False)(image))
    assert torch.isfinite(a).all()

def test_grayscale_and_rgba_use_same_rgb_mapping():
    gray=Image.new('L',(32,32),128)
    assert torch.equal(preprocess(gray,32),preprocess(gray.convert('RGB'),32))
    rgba=Image.new('RGBA',(32,32),(100,120,140,0))
    assert torch.equal(preprocess(rgba,32),preprocess(rgba.convert('RGB'),32))

def test_imagenet_normalization_exact_values():
    output=preprocess(Image.new('RGB',(32,32),(255,0,0)),32)
    expected=torch.tensor([(1-MEAN[0])/STD[0],-MEAN[1]/STD[1],-MEAN[2]/STD[2]])
    assert torch.allclose(output[:,0,0],expected)
