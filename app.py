# app.py
import streamlit as st
import cv2
import time
import numpy as np
import os
import dlib
from ultralytics import YOLO
from datetime import datetime
from dotenv import load_dotenv
import requests
import threading
import pandas as pd

# Load env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/v1/completions"

# Initialize models
st.set_page_config("AI Proctoring", layout="wide")
st.title("🛡️ AI-Based Online Proctoring System")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
topics = st.sidebar.text_input("Quiz Topics (comma-sep)", "Python,Data Structures")
mcq_count = st.sidebar.slider("Number of Questions", 5, 25, 10)
test_dur = st.sidebar.slider("Test Duration (minutes)", 1, 10, 5)
monitor_dur = st.sidebar.slider("Total Monitoring (minutes)", 1, 60, 30)
absent_thresh = st.sidebar.number_input("Absence Threshold (s)", 1, 10, 5)

# Initialize detectors
face_detector = dlib.get_frontal_face_detector()
landmark_model = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
yolo = YOLO("yolov8n.pt")  # auto-downloads if missing

# State
if "logs" not in st.session_state:
    st.session_state.logs = []
if "in_test" not in st.session_state:
    st.session_state.in_test = False

# Helpers
def log_event(evt):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.logs.append({"time":ts, "event":evt})

def trigger_quiz():
    st.session_state.in_test = True
    # call Groq for MCQs
    prompt = (
        f"Generate a {mcq_count}-question multiple-choice quiz on topics: {topics}. "
        "Format as JSON list with question, options, and answer index."
    )
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    data = {"model": "groq-llama3", "prompt": prompt, "max_tokens": 1000}
    resp = requests.post(GROQ_URL, json=data, headers=headers).json()
    quiz = resp.get("choices", [{}])[0].get("text", "[]")
    st.session_state.quiz = pd.read_json(quiz)
    st.session_state.start_test = time.time()
    log_event("Quiz started")

# Main loop
if st.button("Start Proctoring"):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    start = time.time()
    last_face = time.time()
    # schedule quiz every random interval
    def schedule():
        while time.time() - start < monitor_dur*60:
            time.sleep(np.random.randint(60, 120))
            if not st.session_state.in_test:
                trigger_quiz()
    threading.Thread(target=schedule, daemon=True).start()

    while time.time() - start < monitor_dur*60:
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        # 1) Face + gaze/absence detection
        dets = face_detector(rgb, 0)
        if len(dets)==0:
            if time.time()-last_face>absent_thresh:
                log_event("Absence detected")
                st.error("🚨 Candidate absent!")
            else:
                pass
        else:
            last_face = time.time()
            if len(dets)>1:
                log_event("Multiple faces")
                st.error("🚨 Multiple faces!")
            # Gaze: estimate nose direction via landmarks
            shape = landmark_model(rgb, dets[0])
            nose = shape.part(30)
            if nose.x < w*0.3 or nose.x > w*0.7:
                log_event("Looking away")
                st.warning("👀 Looking away!")

        # 2) Mobile detection
        results = yolo.predict(frame, stream=True)
        for r in results:
            for box, cls, _ in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                if int(cls)==67 and float(r.boxes.conf)>.3:  # class 67 = cell phone
                    log_event("Mobile detected")
                    st.error("🚨 Mobile phone detected!")

        # Show frame
        stframe.image(frame, channels="BGR")
        # If in test
        if st.session_state.in_test:
            elapsed = (time.time()-st.session_state.start_test)/60
            if elapsed > test_dur:
                st.session_state.in_test = False
                log_event("Quiz ended")
            else:
                st.subheader("📝 On-going Quiz")
                df = st.session_state.quiz.copy()
                answers = []
                for i,row in df.iterrows():
                    st.write(f"Q{i+1}. {row['question']}")
                    ans = st.radio(f"Select:", row["options"], key=f"q{i}")
                    answers.append(ans)
                if st.button("Submit Quiz"):
                    # score
                    score=0
                    for idx,row in df.iterrows():
                        if answers[idx]==row["answer"]:
                            score+=1
                    log_event(f"Quiz submitted: {score}/{mcq_count}")
                    st.success(f"Score: {score}/{mcq_count}")
                    st.session_state.in_test=False

        time.sleep(0.5)
    cap.release()
    st.success("✅ Monitoring Complete")

    # Download logs
    df_logs = pd.DataFrame(st.session_state.logs)
    st.download_button("📥 Download Log CSV",
                       df_logs.to_csv(index=False),
                       file_name="proctor_logs.csv",
                       mime="text/csv")
