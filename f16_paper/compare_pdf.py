import fitz  # PyMuPDF
from docx import Document
import textract
import difflib
import os

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text
    return text

def extract_text_from_doc(file_path):
    text = textract.process(file_path).decode('utf-8')
    return text

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.docx':
        return extract_text_from_docx(file_path)
    elif ext == '.doc':
        return extract_text_from_doc(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def compare_texts(text1, text2):
    d = difflib.Differ()
    diff = list(d.compare(text1.splitlines(), text2.splitlines()))
    return '\n'.join(diff)

# Exemplo de uso:
file_path1 = '/home/rafael/Mestrado/MGGP_modified_MSSP.doc'
file_path2 = '/home/rafael/Mestrado/2024_IJMIC-137018_PPV (1) (1).docx'

try:
    # Extraia os textos dos arquivos
    text1 = extract_text(file_path1)
    text2 = extract_text(file_path2)

    # Compare os textos
    diff = compare_texts(text1, text2)
    print(diff)
except Exception as e:
    print(f"An error occurred: {e}")
