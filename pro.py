import pandas as pd
from fpdf import FPDF
import fitz  # PyMuPDF
from langchain_community.llms import Ollama
import streamlit as st
import json
import os
import pytesseract
from PIL import Image
from pathlib import Path
import pdfkit

# Initialize Ollama model
llm = Ollama(model="llama3.2:latest", base_url="http://localhost:11434")

# Set output directory for saving converted files
output_dir = Path("documents")
output_dir.mkdir(exist_ok=True)  # Ensure the directory exists

# Function to convert HTML to PDF
path_to_wkhtmltopdf = "C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe"
config = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)
def html_to_pdf_with_pdfkit(html_file, output_pdf):
    try:
        if os.path.exists(html_file):
            pdfkit.from_file(html_file, output_pdf, configuration=config)
            print(f"Successfully converted {html_file} to PDF.")
        else:
            raise FileNotFoundError(f"HTML file {html_file} not found.")
    except Exception as e:
        print(f"Error converting HTML to PDF: {e}")

# Function to convert Excel to PDF
def excel_to_pdf_content_only(excel_file, output_pdf):
    df = pd.read_excel(excel_file)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    
    for index, row in df.iterrows():
        content_line = " ".join([str(cell) for cell in row if pd.notnull(cell)])
        pdf.multi_cell(0, 10, content_line)
        pdf.ln(5)
    
    pdf.output(output_pdf)

# Function to extract text and image information from PDF
def extract_pdf_text(pdf_file):
    doc = fitz.open(pdf_file)
    all_text = []
    
    for page_number, page in enumerate(doc):
        text = page.get_text("text")
        images = len(page.get_images(full=True))
        
        if not text:
            text = ocr_pdf_page(page)
        
        all_text.append({
            "page_number": page_number + 1,
            "content": text,
            "images": images
        })
    
    return all_text

# Function to extract text from PDF using OCR
def ocr_pdf_page(page):
    img = page.get_pixmap()
    pil_image = Image.frombytes("RGB", (img.width, img.height), img.samples)
    text = pytesseract.image_to_string(pil_image)
    return text

# Modified function to parse PDF text into JSON using Ollama
def parse_pdf_to_json(all_text):
    if not all_text:
        return {"error": "No content found in PDF"}

    try:
        # Simple prompt for Ollama
        prompt = "Analyze this PDF content and provide insights:\n"
        for page in all_text:
            prompt += f"Page {page['page_number']}: {page['content']}\n"
        
        # Get Ollama's analysis
        analysis = llm.invoke(prompt)
        
        # Create JSON structure with Ollama's analysis
        pages_json = []
        for page in all_text:
            pages_json.append({
                "page_number": page['page_number'],
                "content": page['content'],
                "images": page['images'],
                "ollama_analysis": analysis
            })
        
        return {"pages": pages_json}
    except Exception as e:
        return {"error": f"Error in Ollama processing: {str(e)}"}

# Function to calculate text and image percentages from parsed JSON content
def calculate_text_and_image_percentage_from_json(json_data):
    page_data_list = []

    for page_data in json_data.get('pages', []):
        page_number = page_data.get('page_number')
        content = page_data.get('content', "")
        images = page_data.get('images', 0)

        text_length = len(content)
        image_count = images
        
        total_content = text_length + image_count * 1000
        text_percentage = (text_length / total_content) * 100 if total_content else 0
        image_percentage = (image_count * 1000 / total_content) * 100 if total_content else 0

        page_data_list.append({
            "page_number": page_number,
            "text_percentage": text_percentage,
            "image_percentage": image_percentage,
            "image_count": image_count
        })
    
    return page_data_list

# Streamlit UI
st.title("Smart Converter 🤖")

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
                excel_to_pdf_content_only(uploaded_file, str(pdf_file))
            elif uploaded_file.type == "text/html":
                html_file_path = output_dir / uploaded_file.name
                with open(html_file_path, "wb") as f:
                    f.write(uploaded_file.read())
                
                html_to_pdf_with_pdfkit(str(html_file_path), str(pdf_file))

            pdf_files.append(pdf_file)

        st.write(f"Converted {len(pdf_files)} files to PDF.")

        # Analyze the PDFs and convert to JSON
        if st.button("Convert PDFs to JSON and Analyze Content"):
            if not pdf_files:
                st.error("No PDF files to analyze.")
            else:
                json_outputs = []
                for pdf_file in pdf_files:
                    all_text = extract_pdf_text(pdf_file)
                    json_output = parse_pdf_to_json(all_text)
                    
                    if "error" in json_output:
                        st.error(f"Error processing {pdf_file}: {json_output['error']}")
                    else:
                        page_data_list = calculate_text_and_image_percentage_from_json(json_output)
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

                        # Save JSON output to file
                        json_file = output_dir / f"{pdf_file.split('.')[0]}_output.json"
                        with open(json_file, "w") as f:
                            json.dump(json_data, f, indent=4)
                        st.write(f"Saved JSON output for {pdf_file} at {json_file}")
else:
    st.write("Upload HTML or Excel files to begin conversion.")
