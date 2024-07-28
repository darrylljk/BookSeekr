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

# Function to rotate the image
def rotate_image(image, angle):
    return image.rotate(angle, expand=True)

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
    found_texts = []
    found_indices = []
    for i in range(len(boxes['text'])):
        word = boxes['text'][i].strip().lower()
        if word:
            similarity = calculate_similarity(word, book_name.lower())
            if similarity >= threshold:
                x, y, w, h = boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i]
                draw.rectangle([x, y, x + w, y + h], outline="red", width=10)
                coords_text = f"({x},{y})"
                draw.text((x, y - 20), coords_text, fill="blue", font=font)  # Append coordinates above bounding box
                draw.text((x, y - 10), word, fill="red", font=font)  # Append text above bounding box
                found_texts.append(word)
                found_indices.append(i)
    return image, bool(found_texts), found_texts, found_indices

st.set_page_config(page_title="BookSeekr", page_icon="📚")
st.title("BookSeekr")

# Introduction to BookSeekr
st.markdown("""
### Welcome to BookSeekr!
BookSeekr helps you quickly locate and identify book titles from a picture of your bookshelf. Simply upload an image, rotate it for correct orientation, and search for a specific word in the book titles.
""")

# Sidebar for user inputs
st.sidebar.header("User Inputs")

# Upload image
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption='Uploaded Image', width=300)

    # Slider and numeric input to select the rotation angle
    st.sidebar.write("Rotate the image to ensure the English words are in the correct orientation.")
    angle = st.sidebar.slider("Rotate image", 0, 360, 0)
    angle_input = st.sidebar.number_input("Or input rotation angle", min_value=0, max_value=360, value=angle)
    
    if angle_input != angle:
        angle = angle_input
    
    rotated_image = rotate_image(uploaded_image, angle)
    st.image(rotated_image, caption='Rotated Image', use_column_width=True)

    book_name = st.sidebar.text_input("Input a single word to search for your book").lower()

    # Validate the book name input
    if " " in book_name:
        st.error("Please enter a single word without spaces.")
    else:
        # Predefined similarity threshold options as buttons
        st.sidebar.subheader("Select Similarity Threshold")
        st.sidebar.write("""
        BookSeekr uses the Levenshtein Similarity Score to measure how similar two words are. 
        It calculates the number of single-character changes needed to turn one word into another. A higher score means the words are more alike.        
        """)
        col1, col2, col3, col4 = st.sidebar.columns(4)
        if col1.button("100%"):
            threshold = 1.0
        elif col2.button("75%"):
            threshold = 0.75
        elif col3.button("50%"):
            threshold = 0.5
        elif col4.button("25%"):
            threshold = 0.25
        else:
            threshold = None

        if threshold is not None:
            boxes = extract_text_and_boxes(rotated_image)
            text = " ".join([word.lower() for word in boxes['text'] if word.strip()])
            # st.write("Extracted Text:")
            # st.write(text)

            # Create a DataFrame to store the extracted text and their bounding boxes
            df = pd.DataFrame({
                'Extracted Text Chunk': boxes['text'],
                'Left': boxes['left'],
                'Top': boxes['top'],
                'Width': boxes['width'],
                'Height': boxes['height'],
                'Confidence': boxes['conf']
            })

            # st.write("Extracted Text DataFrame:")
            # st.write(df.head())

            # Draw bounding boxes around the detected text
            image_with_boxes, found, found_texts, found_indices = draw_boxes(rotated_image, boxes, book_name, threshold)
            st.image(image_with_boxes, caption='Image with detected book name', use_column_width=True)

            if found:
                st.success(f"'{book_name}' found in the image.")
                found_df = pd.DataFrame({
                    'Found Text': [boxes['text'][i] for i in found_indices],
                    'Left': [boxes['left'][i] for i in found_indices],
                    'Top': [boxes['top'][i] for i in found_indices],
                    'Width': [boxes['width'][i] for i in found_indices],
                    'Height': [boxes['height'][i] for i in found_indices],
                    'Confidence': [boxes['conf'][i] for i in found_indices]
                })
                st.write("Found Text DataFrame:")
                st.write(found_df)
            else:
                st.error(f"'{book_name}' not found in the image.")

            # Save the DataFrame to a CSV file
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download CSV", data=csv, file_name='extracted_texts.csv', mime='text/csv')
