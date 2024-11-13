import pandas as pd
from fpdf import FPDF
import fitz  # PyMuPDF for PDF text extraction
from langchain_community.llms import Ollama
import streamlit as st
import json
import os
import pytesseract
from PIL import Image
from pathlib import Path

# Initialize Ollama model
llm = Ollama(model="llama3.1:latest", base_url="http://localhost:11434")

# Set output directory for saving converted files
output_dir = Path("documents")
output_dir.mkdir(exist_ok=True)  # Ensure the directory exists

# Function to convert HTML to PDF
def html_to_pdf(html_file, output_pdf):
    with open(html_file, "r") as file:
        html_content = file.read()
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, html_content)
    pdf.output(output_pdf)

# Function to convert Excel to PDF
def excel_to_pdf(excel_file, output_pdf):
    df = pd.read_excel(excel_file)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Excel to PDF", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=10)
    columns = df.columns.tolist()
    for col in columns:
        pdf.cell(40, 10, col, border=1)
    pdf.ln()
    for _, row in df.iterrows():
        for col in columns:
            pdf.cell(40, 10, str(row[col]), border=1)
        pdf.ln()
    pdf.output(output_pdf)

# Function to extract text from PDF
def extract_pdf_text(pdf_file):
    doc = fitz.open(pdf_file)
    text = ""
    for page in doc:
        text += page.get_text("text")
        if not text:  # If no text is found, try OCR
            text = ocr_pdf_page(page)
    return text

# Function to extract text from PDF using OCR
def ocr_pdf_page(page):
    img = page.get_pixmap()
    pil_image = Image.frombytes("RGB", (img.width, img.height), img.samples)
    text = pytesseract.image_to_string(pil_image)
    return text

# Function to calculate text and image percentages for each page
def calculate_text_and_image_percentage(pdf_file):
    doc = fitz.open(pdf_file)
    page_data_list = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        images = page.get_images(full=True)
        text_length = len(text)
        image_count = len(images)
        
        total_content = text_length + image_count * 1000
        text_percentage = (text_length / total_content) * 100 if total_content else 0
        image_percentage = (image_count * 1000 / total_content) * 100 if total_content else 0
        
        page_data_list.append({
            "page_number": page_num + 1,
            "text_percentage": text_percentage,
            "image_percentage": image_percentage
        })
    return page_data_list

# Function to parse PDF text into JSON using Ollama with enhanced error handling
def parse_pdf_to_json(pdf_text):
    if not pdf_text.strip():
        return {"error": "The PDF content is empty. Skipping JSON conversion."}
    
    prompt = f"""
    You are helping convert a PDF document into a structured JSON format.
    Please output the following content in valid JSON format (without markdown or backticks):

    {{
        "pages": [
            {{
                "page_number": 1,
                "content": "Title, Paragraphs, Tables, Images, etc."
            }}
            ...
        ]
    }}
    
    Here’s the content of the PDF:
    {pdf_text}
    """
    try:
        response = llm.invoke(prompt)
        
        # Check if the response is empty
        if not response:
            return {"error": "The LLM returned an empty response. Unable to parse JSON."}
        
        # Log the raw response for debugging
        print("Raw Response from Ollama:")
        print(response)
        
        # Clean the response by removing any markdown or backticks
        cleaned_response = response.replace("```json", "").replace("```", "").strip()

        # Attempt to parse the response as JSON
        json_data = json.loads(cleaned_response)
        return json_data

    except json.JSONDecodeError as e:
        # Log JSON decoding errors with details
        return {
            "error": f"Error parsing JSON: {str(e)}",
            "content": response
        }
    except Exception as e:
        # Catch all other errors and return a message
        return {
            "error": f"An unexpected error occurred: {str(e)}",
            "content": response
        }

# Streamlit UI
st.title("HTML/Excel to PDF and PDF to JSON Converter")

# Upload HTML, Excel, and PDF files
uploaded_files = st.file_uploader(
    "Upload HTML, Excel, or PDF files (Maximum 5 files)",
    type=["html", "xlsx", "pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) > 5:
        st.error("Please upload no more than 5 files.")
    else:
        pdf_files = []
        for uploaded_file in uploaded_files:
            pdf_file = output_dir / f"{uploaded_file.name.split('.')[0]}.pdf"
            if uploaded_file.type == "application/pdf":
                with open(pdf_file, "wb") as f:
                    f.write(uploaded_file.read())
            elif uploaded_file.type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                excel_to_pdf(uploaded_file, str(pdf_file))
            elif uploaded_file.type == "text/html":
                html_file_path = output_dir / uploaded_file.name
                with open(html_file_path, "wb") as f:
                    f.write(uploaded_file.read())
                html_to_pdf(html_file_path, str(pdf_file))
            pdf_files.append(pdf_file)

        st.write(f"Converted {len(pdf_files)} files to PDF.")

        # Analyze the PDFs and convert to JSON
        if st.button("Convert PDFs to JSON and Analyze Content"):
            if not pdf_files:
                st.error("No PDF files to analyze.")
            else:
                json_outputs = []
                for pdf_file in pdf_files:
                    pdf_text = extract_pdf_text(pdf_file)
                    page_data_list = calculate_text_and_image_percentage(pdf_file)
                    
                    # Parse PDF to JSON for each page
                    json_output = parse_pdf_to_json(pdf_text)
                    
                    if "error" in json_output:
                        st.error(f"Error processing {pdf_file}: {json_output['error']}")
                    else:
                        # Append page-specific text/image percentages
                        json_output["pages_info"] = page_data_list
                        json_outputs.append({pdf_file.name: json_output})

                # Display the JSON results
                for json_output in json_outputs:
                    for pdf_file, json_data in json_output.items():
                        st.write(f"JSON for {pdf_file}:")
                        if isinstance(json_data, dict):
                            st.json(json_data)
                        else:
                            st.error(json_data)
else:
    st.write("Upload HTML or Excel files to begin conversion.")
