import streamlit as st
import os
from ultralytics import YOLO
import shutil

model = YOLO('best.pt')

def remove_and_create_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Deleted Folder: {folder_path}")
    
    else:
        print(f"Folder does not exist: {folder_path}")

    os.mkdir(folder_path)
    print(f"Created Folder: {folder_path}")


def save_uploaded_file(file, folder):
    file_path = os.path.join(folder, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

remove_and_create_folder("runs")
remove_and_create_folder("uploaded_data")

st.title("Tchoukball AI Watcher")

st.header("""
Objective:
The objective of my project is to introduce and promote the sport of tchoukball using an innovative AI model. I believe this AI model is the world's first specifically designed for tchoukball, aimed at increasing awareness and interest in the sport.
""")

# create description to describe project and show link of labelled  dataset
with st.expander("More Details"):
    st.caption("""

    -Data Collection:
    To gather the necessary data, I extensively searched for tchoukball photos on platforms like Google. I captured screenshots of numerous photos and compiled them into a dataset using a website called Roboflow.

    -Data Labelling:
    Next, I meticulously labeled each photo in the dataset, ensuring accurate identification and classification of tchoukball-related elements.

    -Link to my Dataset:
    For easy access, my dataset can be found at the following link: "https://universe.roboflow.com/ai-s8jse/tchouk-track"

    -AI Training:
    To train the AI model, I utilized Google Colab, a powerful platform for machine learning. I employed YOLOv8, a popular object detection algorithm, to train the AI using the labeled photos from my dataset.

    -Deployment:
    For presenting the results, I developed a user-friendly website using Streamlit. This website allows users to upload photos or videos related to tchoukball, and the AI model processes and displays the outcomes using Python.

    By combining data collection, labeling, training, and deployment stages, I aim to create a comprehensive AI solution that promotes tchoukball and engages a wider audience through an interactive web interface.
    """)

# create part to upload image or video
uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "gif", "mp4", "mov"])

if uploaded_file is not None:
    # Save the uploaded file to the "uploaded_data" folder
    file_path = save_uploaded_file(uploaded_file, "uploaded_data")
    st.success(f"File saved successfully: {file_path}")
    
    # Display the uploaded image or video
    if uploaded_file.type.startswith("image"):
        st.image(uploaded_file, caption="Uploaded Image")
    elif uploaded_file.type.startswith("video"):
        st.video(uploaded_file, caption="Uploaded Video")

    # run model on image or video
    source = file_path

    results = model(source,save=True)


    # show the model output from newest folder runs/detect
    model_output = os.listdir("runs/detect")
    model_output = sorted(model_output)
    model_output = model_output[-1]

    final_model_output = os.listdir("runs/detect/" + model_output)[0]

    output_path = "runs/detect/" + model_output + "/" + final_model_output

    print("---------",model_output)

    if uploaded_file.type.startswith("image"):
        st.image(output_path, caption="Uploaded Image")
    elif uploaded_file.type.startswith("video"):
        st.video(output_path, caption="Uploaded Video")


