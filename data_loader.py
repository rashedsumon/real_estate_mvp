"""
data_loader.py
Handles automated downloading and prep of a sample room layout/object 
detection dataset from Hugging Face Hub.
"""
import streamlit as st
from datasets import load_dataset

@st.cache_data(show_spinner="Downloading sample real-estate dataset from Hugging Face...")
def fetch_or_load_dataset():
    """
    Downloads a lightweight, public Hugging Face dataset consisting of indoor scenes.
    For demonstration, we use a subset of a scene-understanding/object detection dataset.
    """
    try:
        # Loading a mini split of 'detection-datasets/coco' or similar light dataset for UI speed
        dataset = load_dataset("detection-datasets/coco", split="train[:5]")
        return dataset
    except Exception as e:
        # Fallback to a standard image dataset if specific splits are unreachable
        st.error(f"Failed to fetch dataset from Hugging Face: {e}")
        return None