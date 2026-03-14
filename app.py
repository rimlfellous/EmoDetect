import streamlit as st
import cv2
import time
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, MaxPooling2D,
    Dropout, Flatten, Dense
)

# -------------------
# Page config
# -------------------

st.set_page_config(
    page_title="EmoDetect",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------
# Custom CSS
# -------------------

st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background-color: #0a0e17;
    background-image: linear-gradient(135deg, #0a0e17 0%, #1a1f2e 50%, #16213e 100%);
}

[data-testid="stSidebar"] {
    background-color: #0f141e;
    background-image: linear-gradient(180deg, #0f141e 0%, #1a1f2e 100%);
}

[data-testid="stHeader"] {
    background-color: #0a0e17 !important;
    background-image: linear-gradient(135deg, #0a0e17 0%, #16213e 100%) !important;
}

[data-testid="stHeader"] * {
    color: #ffffff !important;
}

/* TITLE */

.main-title {
        color: #00d4ff;
        font-size: 3.5rem;
        font-weight: 800;
        text-shadow: 0 0 20px rgba(0,212,255,0.5);
        text-align: center;
        margin-bottom: 1rem;
}

/* SUBTITLE */

.subtitle {
    color: #919ba8;
    text-align: center;
    font-size: 1.3rem;
    font-style: italic;
}

/* CAMERA TITLE */

.camera-title {
    color: white;
    text-align: center;
    font-size: 1.6rem;
    font-weight: 600;
}

/* SIDEBAR */

.sidebar-title {
    color: #e2e8f0;
    font-weight: 600;
}

.sidebar-text {
    color: #b8c2cc;
}

/* BUTTON */

.stButton > button {
    background: linear-gradient(45deg, #00d4ff, #0099cc);
    color: white;
    border-radius: 25px;
    border: none;
    padding: 0.8rem 2rem;
    font-size: 1.1rem;
}

</style>
""", unsafe_allow_html=True)

# -------------------
# Model setup
# -------------------

WEIGHTS_PATH = r"C:\\Users\\Lfellous\\Desktop\\FER2013_results\\emotion_model.weights.h5"

emotion_dict = {
    0: "😡 Angry",
    1: "🤢 Disgusted",
    2: "😱 Fearful",
    3: "😊 Happy",
    4: "😐 Neutral",
    5: "😢 Sad",
    6: "😲 Surprised"
}

EMOTION_COLORS = {
    0: (0,0,255),
    1: (0,128,0),
    2: (255,0,255),
    3: (0,255,255),
    4: (255,255,0),
    5: (255,0,0),
    6: (0,165,255)
}

@st.cache_resource
def load_model():

    model = Sequential([

        Input(shape=(48,48,1)),

        Conv2D(32,(3,3),activation='relu',padding='same'),
        BatchNormalization(),
        Conv2D(32,(3,3),activation='relu',padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(64,(3,3),activation='relu',padding='same'),
        BatchNormalization(),
        Conv2D(64,(3,3),activation='relu',padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(128,(3,3),activation='relu',padding='same'),
        BatchNormalization(),
        Conv2D(128,(3,3),activation='relu',padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(256,(3,3),activation='relu',padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Flatten(),

        Dense(512,activation='relu'),
        BatchNormalization(),
        Dropout(0.4),

        Dense(256,activation='relu'),
        Dropout(0.3),

        Dense(7,activation='softmax')
    ])

    model.load_weights(WEIGHTS_PATH)

    return model


model = load_model()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

prediction_buffer = deque(maxlen=8)

# -------------------
# Emotion prediction
# -------------------

def predict_emotion(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,1.3,5,minSize=(30,30)
    )

    result = frame.copy()

    for (x,y,w,h) in faces:

        roi_gray = gray[y:y+h,x:x+w]

        roi_resized = cv2.resize(roi_gray,(48,48))

        roi_input = roi_resized.astype("float32")/255.0
        roi_input = np.expand_dims(np.expand_dims(roi_input,-1),0)

        raw_pred = model.predict(roi_input,verbose=0)[0]

        prediction_buffer.append(raw_pred)

        smoothed = np.mean(prediction_buffer,axis=0)

        emotion_idx = int(np.argmax(smoothed))
        confidence = smoothed[emotion_idx]*100

        color = EMOTION_COLORS[emotion_idx]

        label = f"{emotion_dict[emotion_idx]} {confidence:.0f}%"

        cv2.rectangle(result,(x,y),(x+w,y+h),color,2)

        cv2.putText(
            result,label,(x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,0.9,color,2
        )

    return result,len(faces)

# -------------------
# Title
# -------------------

st.markdown('<h1 class="main-title">EmoDetect</h1>', unsafe_allow_html=True)

st.markdown(
'<p class="subtitle">Real-time facial emotion recognition powered by CNN</p>',
unsafe_allow_html=True
)

# -------------------
# Sidebar
# -------------------

with st.sidebar:

    st.markdown("### Controls")

    st.success("Model Loaded")

    st.markdown('<p class="sidebar-title">~ Emotion Guide ~</p>', unsafe_allow_html=True)

    for e in emotion_dict.values():
        st.markdown(f'<p class="sidebar-text">{e}</p>', unsafe_allow_html=True)

# -------------------
# Camera section
# -------------------

st.markdown("---")

col1,col2,col3 = st.columns([1,4,1])

with col2:

    st.markdown('<h3 class="camera-title">Live Camera Feed</h3>', unsafe_allow_html=True)

    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False

    start = st.button("Start Emotion Detection",use_container_width=True)
    stop = st.button("Stop Detection",use_container_width=True)

    if start:
        st.session_state.camera_running = True

    if stop:
        st.session_state.camera_running = False

    frame_placeholder = st.empty()
    info_placeholder = st.empty()

    if st.session_state.camera_running:

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Could not open webcam")

        else:

            prediction_buffer.clear()

            while st.session_state.camera_running:

                ret,frame = cap.read()

                if not ret:
                    break

                result,n_faces = predict_emotion(frame)

                result_rgb = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)

                frame_placeholder.image(
                    result_rgb,
                    channels="RGB",
                    use_container_width=True
                )

                info_placeholder.write(f"Faces detected: {n_faces}")

                time.sleep(0.03)

        cap.release()

# -------------------
# Footer
# -------------------

st.markdown("---")

st.markdown(
"<p style='text-align:center;color:#a0bfff;'>Built using Streamlit + TensorFlow | FER2013 dataset</p>",
unsafe_allow_html=True
)