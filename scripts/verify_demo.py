"""Exercise the Streamlit upload/inference/Grad-CAM flow with a real saved model."""
import io
import sys
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from PIL import Image
from streamlit.testing.v1 import AppTest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils import ROOT, save_json
from src.predict import load_predictor, predict_image

def main():
    rows=pd.read_csv(ROOT/'results/split_manifest.csv')
    row=rows[rows.split=='test'].iloc[0]
    with Image.open(ROOT/row.path) as im:
        image=im.convert('RGB');buffer=io.BytesIO();image.save(buffer,format='PNG');buffer.seek(0)
        model,metadata,device=load_predictor()
        expected=predict_image(image,model,metadata,device)
    blank=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    assert not blank.exception
    # AppTest has no native upload setter; inject only file input, keeping model/Grad-CAM real.
    with patch('streamlit.file_uploader',return_value=buffer):
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    assert not app.exception, [item.message for item in app.exception]
    assert app.subheader[0].value==expected['predicted_class']
    assert app.metric[0].value==f"{expected['confidence']:.1%}"
    assert app.checkbox[0].value is True
    assert len(app.dataframe[0].value)==6
    import json
    low_path=json.loads((ROOT/'results/error_examples.json').read_text())['low_confidence'][0]['path']
    with Image.open(ROOT/low_path) as low_image:
        low_buffer=io.BytesIO();low_image.convert('RGB').save(low_buffer,format='PNG');low_buffer.seek(0)
    with patch('streamlit.file_uploader',return_value=low_buffer):
        low_app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    assert not low_app.exception
    assert any(item.value=='Prediction confidence is low. Try a clearer image of a single apple leaf.' for item in low_app.warning)
    report={'low_confidence_warning_verified':True,'empty_upload_app_passed':True,'uploaded_image_flow_passed':True,
        'inference_matches_direct_api':True,'gradcam_executed':True,'ranked_class_rows':6,
        'input_injection':'File uploader return value patched because AppTest has no native upload setter; inference and Grad-CAM executed normally.',
        'tested_image':row.path,'prediction':expected['predicted_class'],'confidence':expected['confidence']}
    save_json(ROOT/'results/demo_verification.json',report);print(report)

if __name__=='__main__':main()
