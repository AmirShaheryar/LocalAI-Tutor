import fitz  
import json
import re
import os

MATH_REGEX = re.compile(r'(\$\$?[\s\S]+?\$\$?|\\\(.*?\\\)|\\\[.*?\\\]|\\int|\\frac|\\sum|\\sqrt)')
MATH_FONTS = ["math", "cmsy", "cmex", "symbol", "stix", "cambriamath"]
UNICODE_MATH = ["∫", "∑", "∏", "√", "∂", "∇", "≤", "≥", "≠", "±", "≈", "∞", "α", "β", "θ", "π"]

def is_math_span(text, font_name=""):
    """Determines if a string or font name indicates mathematical notation."""
    if MATH_REGEX.search(text):
        return True
    if any(kw in font_name.lower() for kw in MATH_FONTS):
        return True
    if any(char in text for char in UNICODE_MATH):
        return True
    return False


def parse_pdf(pdf_path, output_img_dir="data/extracted_images"):
    """
    Extracts text blocks, page numbers, math expressions, and embedded images
    from a PDF file and structures them into a clean dictionary.
    """
    os.makedirs(output_img_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    
    structured_data = {
        "document": os.path.basename(pdf_path),
        "total_pages": len(doc),
        "pages": []
    }

    for page_num, page in enumerate(doc, start=1):
        page_data = {
            "page_number": page_num,
            "content_blocks": [],
            "extracted_images": []
        }

        page_dict = page.get_text("dict")
        for block in page_dict["blocks"]:
            if block["type"] == 0:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue
                        
                        is_math = is_math_span(text, span["font"])
                        
                        page_data["content_blocks"].append({
                            "type": "math_formula" if is_math else "text",
                            "content": text,
                            "font": span["font"],
                            "size": round(span["size"], 1),
                            "is_latex": is_math
                        })

        for img_idx, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            base_img = doc.extract_image(xref)
            img_filename = f"page_{page_num}_img_{img_idx}.{base_img['ext']}"
            img_path = os.path.join(output_img_dir, img_filename)
            
            with open(img_path, "wb") as f:
                f.write(base_img["image"])
            
            page_data["extracted_images"].append(img_path)

        structured_data["pages"].append(page_data)

    doc.close()
    return structured_data


if __name__ == "__main__":

    current_script_path = os.path.abspath(__file__)
    parsers_dir = os.path.dirname(current_script_path)
    src_dir = os.path.dirname(parsers_dir)
    project_root = os.path.dirname(src_dir)
    
    # Construct exact absolute path to the PDF
    sample_pdf = os.path.join(project_root, "data", "raw_pdfs", "sample_fonts_test.pdf")
    output_file = os.path.join(project_root, "outputs", "day1_structured_output.json")
    
    print(f" Searching for file at: {sample_pdf}")
    
    if os.path.exists(sample_pdf):
        result = parse_pdf(sample_pdf)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        print(f" Success! Structured JSON saved to '{output_file}'")
    else:
        print(f" File still not found at: '{sample_pdf}'")