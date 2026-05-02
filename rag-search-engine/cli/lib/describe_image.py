import os
import mimetypes
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)
model = 'gemma-3-27b-it'

def describe_image_command(image_path: str, query: str) -> str:
    mime, _ = mimetypes.guess_type(arg.simag)
