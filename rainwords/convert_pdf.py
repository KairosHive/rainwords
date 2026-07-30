import os
from pathlib import Path

# Shared sanitization pipeline (also used by the live upload endpoint).
from .text_pipeline import clean_text, extract_pdf_text

# Project root = parent of the `rainwords` package
BASE_DIR = Path(__file__).resolve().parent.parent

# Where your PDFs are and where .txt files will go
INPUT_FOLDER = BASE_DIR / "corpuses"
OUTPUT_FOLDER = BASE_DIR / "corpuses"


def convert_single_pdf(pdf_path: str, txt_path: str):
    print(f"\n--- Converting: {os.path.basename(pdf_path)} ---")
    try:
        raw_text = extract_pdf_text(pdf_path)
    except Exception as e:
        print(f"  ❌ Could not open PDF: {e}")
        return

    if not raw_text.strip():
        print("  ⚠ No text extracted from this PDF, skipping.")
        return

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

        # Skip if .txt already exists
        if txt_path.exists():
            print(f"  ⏩ Skipping {pdf_name}: TXT already exists.")
            continue

        convert_single_pdf(str(pdf_path), str(txt_path))


def main():
    """Entry point for the rainwords.convert_pdf CLI."""
    batch_convert_pdfs()
    print("\nDone. You can now run `rainwords.corpus_builder` to rebuild your FAISS index.")


if __name__ == "__main__":
    main()
