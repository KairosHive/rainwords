import os
import re
from pypdf import PdfReader

# --- Configuration ---
INPUT_FOLDER = "../corpuses"   # where your PDFs are
OUTPUT_FOLDER = "../corpuses"  # where .txt files will go (can be same as input)


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
    script_dir = os.path.dirname(__file__)
    input_dir = os.path.join(script_dir, INPUT_FOLDER)
    output_dir = os.path.join(script_dir, OUTPUT_FOLDER)

    if not os.path.exists(input_dir):
        print(f"Input folder not found: {input_dir}")
        return

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in: {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF(s) in '{INPUT_FOLDER}'")

    for pdf_name in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_name)
        base, _ = os.path.splitext(pdf_name)
        txt_name = base + ".txt"      # same name as PDF, with .txt
        txt_path = os.path.join(output_dir, txt_name)

        convert_single_pdf(pdf_path, txt_path)


if __name__ == "__main__":
    batch_convert_pdfs()
    print("\nDone. You can now run `python build_index.py` to rebuild your FAISS index.")
