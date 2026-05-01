import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not set in .env")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model='gemma-3-27b-it',
    contents='"Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."'
)

print(response.text)
metadata = response.usage_metadata
print(f"Prompt tokens: {metadata.prompt_token_count}")
print(f"Response tokens: {metadata.candidates_token_count}")
