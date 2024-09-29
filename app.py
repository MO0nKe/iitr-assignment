import os
import spaces
import re
import time

import streamlit as st
import torch

from PIL import Image

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


@st.cache_resource(show_spinner=False)
def load_model():

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct").eval()

    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", 
                                              trust_remote_code=True, 
                                              min_pixels=min_pixels, 
                                              max_pixels=max_pixels)

    if device != model.device:
        model.to(device)

    return model, processor


@st.cache_data(show_spinner=False)
def extract_text_from_image(image_path, _model, _processor):
    image = Image.open(image_path)

    text_query = "Transcribe the text in the image. Return in JSON format"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": text_query},
            ],
        },
        {
            "role": "assistant",
            "content": '{"Transcribed Text": "'
        },
    ]

    text = _processor.apply_chat_template(
        messages, tokenize=False, continue_final_message=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )   

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    inputs.to(device)
    

    generated_ids = _model.generate(**inputs, max_new_tokens=1000)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = _processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    if output_text:
        output_text = output_text[0]
        output_text = output_text.replace('"}', "")

    return output_text

def highlight_text(text, word, color="lightgreen"):
    # Use a case-insensitive regex to match the word
    pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
    
    # Wrap the matched word with <mark> tag for highlighting
    highlighted_text = pattern.sub(f"<span style='background-color:{color}'>{word}</span>", text)
    
    # Return the highlighted text
    return highlighted_text

def main():

    with st.spinner("Loading the model..."):
        model, processor = load_model()

    st.title("AI-Assignment: Image OCR")


    uploaded_image = st.file_uploader("Upload an image", type=["jpeg", "jpg", "png"])

    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Extract text using your chosen model
        extracted_text = extract_text_from_image(uploaded_image, model, processor)
        
        if not extracted_text:
            st.error("No text detected. Please upload an image with recognizable text.")
        else:
            st.subheader("Extracted Text:")
            st.write(extracted_text)

        keyword = st.text_input("Enter a keyword to search")
        if keyword:
            # Highlight the word in the extracted text
            highlighted_text = highlight_text(extracted_text, keyword)
            
            # Display the highlighted text using st.markdown to allow HTML rendering
            st.markdown(highlighted_text, unsafe_allow_html=True)


if __name__ == '__main__':
    main()