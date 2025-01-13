from PyPDF2 import PdfReader

class PDFLoader:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_text(self):
        reader = PdfReader(self.pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
