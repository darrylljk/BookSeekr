import os
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import cv2
import numpy as np
import Levenshtein
import pandas as pd

# Ensure pytesseract knows the location of the tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to load the image from a directory
def load_image(image_path):
    if os.path.exists(image_path):
        image = Image.open(image_path)
        return image
    else:
        st.error("Image not found.")
        return None

# Function to preprocess the image for better OCR results
def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((1, 1), np.uint8)
    processed_image = cv2.dilate(binary, kernel, iterations=1)
    processed_image = cv2.erode(processed_image, kernel, iterations=1)
    return processed_image

# Function to extract text and bounding boxes from the image using OCR
def extract_text_and_boxes(image):
    processed_image = preprocess_image(image)
    config = "--psm 6"  # Use PSM mode 6 for more structured text extraction
    data = pytesseract.image_to_data(processed_image, config=config, output_type=pytesseract.Output.DICT)
    return data

# Function to calculate similarity score between two strings
def calculate_similarity(s1, s2):
    return Levenshtein.ratio(s1, s2)

# Function to draw bounding boxes and append text
def draw_boxes(image, boxes, book_name, threshold):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    found = False
    for i in range(len(boxes['text'])):
        word = boxes['text'][i].strip().lower()
        if word:
            similarity = calculate_similarity(word, book_name.lower())
            if similarity >= threshold:
                found = True
                x, y, w, h = boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i]
                draw.rectangle([x, y, x + w, y + h], outline="red", width=8)
                draw.text((x, y - 10), word, fill="red", font=font)  # Append text above bounding box
    return image, found

st.set_page_config(page_title="BookSeekr", page_icon="📚")
st.title("BookSeekr")

# Explanation of Levenshtein similarity
st.markdown("""
#### Levenshtein Similarity Score
The Levenshtein similarity score measures how similar two strings are by calculating the minimum number of single-character edits required to change one word into the other. A higher score indicates more similarity.
""")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    book_name = st.text_input("Input a single word to search for your book").lower()
    
    # Validate the book name input
    if " " in book_name:
        st.error("Please enter a single word without spaces.")
    else:
        # Predefined similarity threshold options
        threshold_options = {
            "Perfect Match (100%)": 1.0,
            "Highly Similar (75%)": 0.75,
            "Moderately Similar (50%)": 0.5,
            "Less Similar (25%)": 0.25
        }
        
        threshold_label = st.selectbox("Select Similarity Threshold", list(threshold_options.keys()))
        threshold = threshold_options[threshold_label]

        if st.button("Find Book"):
            boxes = extract_text_and_boxes(uploaded_image)
            text = " ".join([word.lower() for word in boxes['text'] if word.strip() and boxes['conf'][boxes['text'].index(word)] != -1])
            # st.write("Extracted Text:")
            # st.write(text)

            # Filter out text with confidence -1
            valid_indices = [i for i, conf in enumerate(boxes['conf']) if conf != -1]
            filtered_boxes = {key: [boxes[key][i] for i in valid_indices] for key in boxes.keys()}

            # Create a DataFrame to store the extracted text and their bounding boxes
            df = pd.DataFrame({
                'Extracted Text Chunk': filtered_boxes['text'],
                'Left': filtered_boxes['left'],
                'Top': filtered_boxes['top'],
                'Width': filtered_boxes['width'],
                'Height': filtered_boxes['height'],
                'Confidence': filtered_boxes['conf']
            })

            st.write("Extracted Text DataFrame:")
            st.write(df.head())

            # Draw bounding boxes around the detected text
            image_with_boxes, found = draw_boxes(uploaded_image, filtered_boxes, book_name, threshold)
            st.image(image_with_boxes, caption='Image with detected book name', use_column_width=True)

            if found:
                st.success(f"'{book_name}' found in the image.")
            else:
                st.error(f"'{book_name}' not found in the image.")

            # Save the DataFrame to a CSV file
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download CSV", data=csv, file_name='extracted_texts.csv', mime='text/csv')
