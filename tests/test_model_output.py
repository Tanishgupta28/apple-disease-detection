"""Architecture, freezing, inference serialization, and Grad-CAM contracts."""
import pytest
import torch
from PIL import Image
from src.models import build_model, configure_phase, training_mode
from src.gradcam import gradcam
from src.predict import load_predictor, predict_image

torch.set_num_threads(2)

@pytest.mark.parametrize('name',['custom_cnn','mobilenetv2','resnet50'])
def test_six_class_output(name):
    model=build_model(name,pretrained=False).eval()
    with torch.inference_mode():logits=model(torch.randn(2,3,64,64))
    assert logits.shape==(2,6)
    assert torch.isfinite(logits).all()
    assert torch.allclose(logits.softmax(1).sum(1),torch.ones(2))

@pytest.mark.parametrize('name',['mobilenetv2','resnet50'])
def test_transfer_learning_freezes_only_expected_parameters(name):
    model=build_model(name,pretrained=False)
    configure_phase(model,name,False)
    frozen_trainable={n for n,p in model.named_parameters() if p.requires_grad}
    prefix='classifier' if name=='mobilenetv2' else 'fc'
    assert frozen_trainable and all(n.startswith(prefix) for n in frozen_trainable)
    configure_phase(model,name,True)
    tuned_trainable={n for n,p in model.named_parameters() if p.requires_grad}
    assert frozen_trainable < tuned_trainable
    assert len(tuned_trainable)<len(list(model.named_parameters()))
    training_mode(model,name)
    assert all(not module.training for module in model.modules() if isinstance(module,torch.nn.BatchNorm2d))

def test_checkpoint_preserves_class_order_and_inference(tmp_path):
    classes=['Apple Scab','Rust','Frogeye Leaf Spot','Powdery Mildew','Alternaria Leaf Spot','Healthy']
    model=build_model('custom_cnn',pretrained=False).eval()
    with torch.no_grad():
        model.classifier[-1].weight.zero_();model.classifier[-1].bias.copy_(torch.arange(6,dtype=torch.float32))
    metadata={'architecture':'custom_cnn','image_size':64,'classes':classes}
    checkpoint=tmp_path/'model.pth';torch.save({'state_dict':model.state_dict(),'metadata':metadata},checkpoint)
    loaded,saved,device=load_predictor(checkpoint,device='cpu')
    result=predict_image(Image.new('RGB',(40,40)),loaded,saved,device)
    assert result['predicted_class']=='Healthy'
    assert list(result['probabilities'])==classes
    assert sum(result['probabilities'].values())==pytest.approx(1.)

@pytest.mark.parametrize('name',['custom_cnn','mobilenetv2','resnet50'])
def test_gradcam_finite_shape_and_hooks_released(name):
    model=build_model(name,pretrained=False).eval()
    before=sum(len(m._forward_hooks) for m in model.modules())
    heatmap,index=gradcam(model,name,torch.randn(1,3,64,64))
    assert heatmap.shape==(64,64)
    assert 0<=index<6
    assert (heatmap>=0).all() and (heatmap<=1).all()
    assert sum(len(m._forward_hooks) for m in model.modules())==before
