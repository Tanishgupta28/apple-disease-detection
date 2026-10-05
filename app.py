"""Streamlit apple leaf classification and Grad-CAM demonstration."""
from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st
from src.data import preprocess
from src.gradcam import gradcam, overlay
from src.predict import load_predictor, predict_image
from src.utils import load_config

st.set_page_config(page_title='Apple Leaf Disease Detection', page_icon='🍎', layout='centered')
st.title('Apple Leaf Disease Detection')
st.write('Upload a clear photograph of one apple leaf to explore six-class image classification.')
st.caption('An NIT internship learning project. Predictions can be wrong; the model has no unknown-image class.')

@st.cache_resource
def predictor():
    """Load the selected checkpoint once per app process."""
    return load_predictor()

try:
    model, metadata, device = predictor()
except FileNotFoundError as error:
    st.info(str(error))
    st.caption('See the README for training or downloading the released model.')
    st.stop()

uploaded=st.file_uploader('Apple leaf image',type=['jpg','jpeg','png','webp'])
if uploaded is not None:
    try:
        image=ImageOps.exif_transpose(Image.open(uploaded)).convert('RGB')
        image.load()
    except (UnidentifiedImageError,OSError):
        st.error('The uploaded file could not be read as an image.');st.stop()
    st.image(image,caption='Uploaded apple leaf',width='stretch')
    result=predict_image(image,model,metadata,device)
    st.subheader(result['predicted_class'])
    st.metric('Model confidence',f"{result['confidence']:.1%}")
    if result['confidence']<load_config()['low_confidence_threshold']:
        st.warning('Prediction confidence is low. Try a clearer image of a single apple leaf.')
    st.dataframe([{'Disease':row['class'],'Probability':f"{row['probability']:.2%}"}
                  for row in result['top_predictions']],hide_index=True,width='stretch')
    st.caption('Softmax confidence is not calibrated diagnostic certainty.')
    if st.checkbox('Show Grad-CAM',value=True):
        with st.spinner('Calculating Grad-CAM…'):
            tensor=preprocess(image,metadata['image_size']).unsqueeze(0).to(device)
            heatmap,index=gradcam(model,metadata['architecture'],tensor)
            st.image(overlay(image,heatmap),caption='Grad-CAM: regions associated with the predicted class',width='stretch')
            st.caption('Inspect this visualization for attention on lesions or backgrounds. It does not prove a biological diagnosis.')
