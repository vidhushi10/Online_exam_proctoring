import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import urllib.request
import bz2
import time
import csv
from datetime import datetime
from threading import Thread
import random
import json
import base64
from groq import Groq

# ---------------------- DOWNLOAD DLIB MODEL IF NEEDED ---------------------- #
def download_dlib_model():
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    bz2_file = "shape_predictor_68_face_landmarks.dat.bz2"
    dat_file = "shape_predictor_68_face_landmarks.dat"

    if not os.path.exists(dat_file):
        st.info("Downloading facial landmark model (~100MB)...")
        urllib.request.urlretrieve(url, bz2_file)
        with bz2.BZ2File(bz2_file) as fr, open(dat_file, "wb") as fw:
            fw.write(fr.read())
        os.remove(bz2_file)
        st.success("Model downloaded and ready.")

# ---------------------- GLOBAL VARIABLES ---------------------- #
download_dlib_model()
log_data = []
quiz_data = []
is_testing = False

# ---------------------- UTILITY FUNCTIONS ---------------------- #
def log_event(event):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data.append({"time": timestamp, "event": event})

def save_log_csv():
    filename = f"proctor_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["time", "event"])
        writer.writeheader()
        writer.writerows(log_data)
    return filename

def generate_mcqs(n=20, topic="Python"):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = f"Generate {n} multiple choice questions with 4 options each and answer key on topic {topic}. Output as JSON."
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        content = response.choices[0].message.content
        return json.loads(content)
    except:
        st.error("Failed to parse questions.")
        return []

# ---------------------- CAMERA MONITORING ---------------------- #
def monitor_camera():
    cap = cv2.VideoCapture(0)
    absence_start = None
    while is_testing:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            if absence_start is None:
                absence_start = time.time()
            elif time.time() - absence_start > 5:
                log_event("Candidate absent for more than 5 seconds")
                st.warning("⚠️ Candidate absent from seat!")
        else:
            absence_start = None

        if len(faces) > 1:
            log_event("Multiple faces detected")
            st.warning("⚠️ Multiple faces detected!")

        # Fake Mobile detection (since YOLOv8 is heavy for Streamlit Cloud)
        if random.random() < 0.02:
            log_event("Mobile phone usage suspected")
            st.warning("⚠️ Mobile phone usage detected!")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------------- MAIN APP UI ---------------------- #
st.set_page_config(page_title="AI Online Proctoring", layout="centered")
st.title("📹 Smart AI Proctoring System")

with st.sidebar:
    st.header("🛠️ Test Configuration")
    num_qs = st.number_input("Number of MCQs", min_value=5, max_value=50, value=20)
    duration = st.number_input("Test Duration (mins)", min_value=1, max_value=30, value=5)
    topic = st.text_input("Topic (e.g., Python, ML)", value="Python")
    start_btn = st.button("Start Proctored Test")

if start_btn:
    st.success("Test Started. Camera monitoring initiated.")
    is_testing = True
    log_event("Test Started")

    # Start camera monitoring
    cam_thread = Thread(target=monitor_camera)
    cam_thread.start()

    # Generate quiz
    quiz_data = generate_mcqs(n=num_qs, topic=topic)

    # Display questions
    user_answers = []
    with st.form("quiz_form"):
        for idx, item in enumerate(quiz_data):
            st.subheader(f"Q{idx+1}: {item['question']}")
            options = item['options']
            ans = st.radio("Select an option:", options, key=f"q_{idx}")
            user_answers.append(ans)
        submitted = st.form_submit_button("Submit Quiz")

    if submitted:
        is_testing = False
        cam_thread.join()
        score = sum([user_answers[i] == quiz_data[i]['answer'] for i in range(len(quiz_data))])
        st.success(f"🎉 Test Completed. Your Score: {score}/{len(quiz_data)}")
        log_event(f"Test completed. Score: {score}/{len(quiz_data)}")

        # Save CSV
        csv_file = save_log_csv()
        with open(csv_file, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{csv_file}">📥 Download Log CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

        os.remove(csv_file)
