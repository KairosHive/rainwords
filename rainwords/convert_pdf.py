import os
import re
from pypdf import PdfReader
from pathlib import Path

# Project root = parent of the `rainwords` package
BASE_DIR = Path(__file__).resolve().parent.parent

# Where your PDFs are and where .txt files will go
INPUT_FOLDER = BASE_DIR / "corpuses"
OUTPUT_FOLDER = BASE_DIR / "corpuses"


def clean_text(raw_text: str) -> str:
    """
    Clean and lightly reformat extracted text:
    - de-hyphenate words broken across lines
    - remove pure page-number lines
    - rebuild paragraph breaks into double newlines
    """
    # 1. De-hyphenate words broken across lines: "nev-\ner" -> "never"
    raw_text = re.sub(r'-\n([a-zA-Z])', r'\1', raw_text)

    # 2. Split into lines to filter junk
    lines = raw_text.split('\n')
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # Skip completely empty lines
        if not stripped:
            continue

        # Skip pure numeric or simple roman numeral lines (page numbers)
        if re.fullmatch(r'\d+', stripped):
            continue
        if re.fullmatch(r'[ivxlcdmIVXLCDM]+', stripped):
            continue

        # If you want, you can add more header/footer patterns here.
        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)

    # 3. Paragraph logic:
    #    - multiple newlines -> paragraph break marker
    #    - sentence-ending newline -> paragraph break marker
    #    - remaining single newlines -> spaces (soft wraps)
    text = re.sub(r'(\n\s*){2,}', '<<PARAGRAPH_BREAK>>', text)
    text = re.sub(r'([.?!"])\n', r'\1<<PARAGRAPH_BREAK>>', text)
    text = re.sub(r'\n', ' ', text)

    # 4. Turn markers back into double newlines
    text = re.sub(r'\s*<<PARAGRAPH_BREAK>>\s*', '\n\n', text).strip()

    return text


def convert_single_pdf(pdf_path: str, txt_path: str):
    print(f"\n--- Converting: {os.path.basename(pdf_path)} ---")
    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        print(f"  ❌ Could not open PDF: {e}")
        return

    all_page_text = []

    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
            if text:
                all_page_text.append(text)
            else:
                print(f"  ⚠ Page {i+1}: no text extracted.")
        except Exception as e:
            print(f"  ⚠ Page {i+1}: error extracting text: {e}")

    if not all_page_text:
        print("  ⚠ No text extracted from this PDF, skipping.")
        return

    raw_text = "\n".join(all_page_text)
    final_text = clean_text(raw_text)

    # Ensure output folder exists
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)

    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(final_text)
        print(f"  ✅ Saved to: {txt_path}")
    except Exception as e:
        print(f"  ❌ Error writing TXT: {e}")


def batch_convert_pdfs():
    input_dir = INPUT_FOLDER
    output_dir = OUTPUT_FOLDER

    if not input_dir.exists():
        print(f"Input folder not found: {input_dir}")
        return

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in: {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF(s) in '{input_dir}'")

    for pdf_name in pdf_files:
        pdf_path = input_dir / pdf_name
        base, _ = os.path.splitext(pdf_name)
        txt_name = base + ".txt"
        txt_path = output_dir / txt_name

        convert_single_pdf(str(pdf_path), str(txt_path))



def main():
    """Entry point for the rainwords.convert_pdf CLI."""
    batch_convert_pdfs()
    print("\nDone. You can now run `rainwords.corpus_builder` to rebuild your FAISS index.")


if __name__ == "__main__":
    main()

