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