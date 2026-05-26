import sys
import os

try:
    import PyPDF2
    def extract_text(pdf_path):
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
except ImportError:
    def extract_text(pdf_path):
        return "PyPDF2 not installed. Use 'pip install PyPDF2' if needed."

pdf_path = r"h:\AdminFiles\Desktop\New folder\DeepAir\Docs\Proposal\Ml-project-proposal.pdf"
if os.path.exists(pdf_path):
    print(extract_text(pdf_path))
else:
    print(f"File not found: {pdf_path}")
