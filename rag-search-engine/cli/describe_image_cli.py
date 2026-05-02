import argparse
import os
import mimetypes

from lib.describe_image import describe_image_command

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)
model = 'gemini-2.5-flash'


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite a query based on the contents of an image"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image file",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Text query to rewrite based on the image",
    )

    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, "rb") as f:
        image_data = f.read()

        prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
        - Synthesize visual and textual information
        - Focus on movie-specific details (actors, scenes, style, etc.)
        - Return only the rewritten query, without any additional commentary
        """

        parts = [
            prompt,
            types.Part.from_bytes(data=image_data, mime_type=mime),
            args.query.strip()
        ]

        response = client.models.generate_content(
            model=model,
            contents=parts,
        )

        print(f"Rewritten query: {response.text.strip()}")
        if response.usage_metadata is not None:
            print(f"Total tokens:    {response.usage_metadata.total_token_count}")



if __name__ == "__main__":
    main()
