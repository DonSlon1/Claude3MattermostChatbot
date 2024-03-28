import base64
import logging
import traceback
from io import BytesIO
from pdf2image import convert_from_bytes
import requests


def process_pdf(file_content):
    try:
        # Convert PDF to images
        images = convert_from_bytes(file_content.getvalue())

        # Convert images to base64 and create messages
        image_messages = []
        for image in images:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_messages.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_base64,
                },
            })

        return image_messages

    except Exception as e:
        logging.error(f"Error processing PDF: {str(e)} {traceback.format_exc()}")
        return None
