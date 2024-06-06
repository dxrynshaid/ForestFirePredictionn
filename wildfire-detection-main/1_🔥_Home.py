import streamlit as st 
import cv2
from ultralytics import YOLO
import requests 
from PIL import Image
import os
from glob import glob
from numpy import random
import io

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model


def predict_image(model, image, conf_threshold, iou_threshold):
    
    res = model.predict(
        image,
        conf=conf_threshold,
        iou=iou_threshold,
        device='cpu',
    )
    
    class_name = model.model.names
    classes = res[0].boxes.cls
    class_counts = {}
    
    
    for c in classes:
        c = int(c)
        class_counts[class_name[c]] = class_counts.get(class_name[c], 0) + 1

   
    prediction_text = 'Predicted '
    for k, v in sorted(class_counts.items(), key=lambda item: item[1], reverse=True):
        prediction_text += f'{v} {k}'
        
        if v > 1:
            prediction_text += 's'
        
        prediction_text += ', '

    prediction_text = prediction_text[:-2]
    if len(class_counts) == 0:
        prediction_text = "No objects detected"

    
    latency = sum(res[0].speed.values())  
    latency = round(latency / 1000, 2)
    prediction_text += f' in {latency} seconds.'

 
    res_image = res[0].plot()
    res_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)
    
    return res_image, prediction_text

def main():
    st.set_page_config(
        page_title="Wildfire Detection",
        page_icon="🔥",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
        .container {
            max-width: 800px;
        }
        .title {
            text-align: center;
            font-size: 35px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .description {
            margin-bottom: 30px;
        }
        .instructions {
            margin-bottom: 20px;
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    
    st.markdown("<div class='title'>Forest Fire Detection</div>", unsafe_allow_html=True)

    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.write("")
    with col2:
        logos = glob('dalle-logos/*.png')
        if logos:
            logo = random.choice(logos)
            st.image(logo, use_column_width=True, caption="Shaidarov Daryn\nToqsanbai Abylkair\nBDA-2202")
            st.sidebar.image(logo, use_column_width=True, caption="Shaidarov Daryn\nToqsanbai Abylkair\nBDA-2202")
        else:
            st.warning("No logos found in 'dalle-logos' directory.")
    with col3:
        st.write("")

   
    st.markdown(
    """
    <div style='text-align: center;'>
        <h2>🔥 <strong>Forest Fire Detection App</strong></h2>
        <p>Experience the effectiveness of our detection model by uploading an image or providing a URL.</p>
    </div>
    """,
    unsafe_allow_html=True
)

    
    st.markdown("---")

    
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.radio("Select Model Type", ("Fire Detection", "General"), index=0)

    models_dir = "general-models" if model_type == "General" else "fire-models"
    
    st.write(f"Looking for models in: {models_dir}")
    if not os.path.exists(models_dir):
        st.error(f"Directory '{models_dir}' does not exist. Please make sure the directory and model files are in place.")
        return

    model_files = [f.replace(".pt", "") for f in os.listdir(models_dir) if f.endswith(".pt")]
    st.write(f"Found model files: {model_files}")
    if not model_files:
        st.error(f"No model files found in directory '{models_dir}'. Please add the necessary model files.")
        return
    
    with col2:
        selected_model = st.selectbox("Select Model Size", sorted(model_files), index=0)

    
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("What is General?"):
            st.caption("The General model is an additional model that was added for demonstration purposes.")
            st.caption("It is pre-trained on the COCO dataset, which consists of various objects across 80 classes.")
            st.caption("Please note that this model may not be optimized specifically for fire detection.")
            st.caption("For accurate fire and smoke detection, it is recommended to choose the Fire Detection model.")
            st.caption("The General model can be used to detect objects commonly found in everyday scenes.")
    
    with col2:
        with st.expander("Size Information"):
            st.caption("Models are available in different sizes, indicated by n, s, m, and l.")
            st.caption("- n: Nano")
            st.caption("- s: Small")
            st.caption("- m: Medium")
            st.caption("- l: Large")
            st.caption("The larger the model, the more precise the detections, but the slower the inference time.")
            st.caption("On the other hand, smaller models are faster but may sacrifice some precision.")
            st.caption("Choose a model based on the trade-off between speed and precision that best suits your needs.")

   
    model_path = os.path.join(models_dir, selected_model + ".pt")
    st.write(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' does not exist. Please make sure the file is in place.")
        return

    model = load_model(model_path)

    
    st.markdown("---")

   
    col1, col2 = st.columns(2)
    with col2:
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.20, 0.05)
        with st.expander("What is Confidence Threshold?"):
            st.caption("The Confidence Threshold is a value between 0 and 1.")
            st.caption("It determines the minimum confidence level required for an object detection.")
            st.caption("If the confidence of a detected object is below this threshold, it will be ignored.")
            st.caption("You can adjust this threshold to control the number of detected objects.")
            st.caption("Lower values make the detection more strict, while higher values allow more detections.")
    with col1:
        iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.5, 0.05)
        with st.expander("What is IOU Threshold?"):
            st.caption("The IOU (Intersection over Union) Threshold is a value between 0 and 1.")
            st.caption("It determines the minimum overlap required between the predicted bounding box")
            st.caption("and the ground truth box for them to be considered a match.")
            st.caption("You can adjust this threshold to control the precision and recall of the detections.")
            st.caption("Higher values make the matching more strict, while lower values allow more matches.")

    
    st.markdown("---")

    
    image = None
    image_source = st.radio("Select image source:", ("Enter URL", "Upload from Computer"))
    if image_source == "Upload from Computer":
        
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
        else:
            image = None

    else:
        
        url = st.text_input("Enter the image URL:")
        if url:
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()  
                image = Image.open(response.raw)
            except requests.exceptions.RequestException as e:
                st.error(f"Error loading image from URL: {e}")
                image = None

    if image:
        
        with st.spinner("Detecting"):
            try:
                prediction, text = predict_image(model, image, conf_threshold, iou_threshold)
                st.image(prediction, caption="Prediction", use_column_width=True)
                st.success(text)
                
                prediction = Image.fromarray(prediction)
                image_buffer = io.BytesIO()
                prediction.save(image_buffer, format='PNG')
                st.download_button(
                    label='Download Prediction',
                    data=image_buffer.getvalue(),
                    file_name='prediction.png',
                    mime='image/png'
                )
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        
if __name__ == "__main__":
    main()
