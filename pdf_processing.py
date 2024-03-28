import base64
import logging
import traceback
from io import BytesIO
from pdf2image import convert_from_bytes


def process_pdf(file_content, file_name=""):
    try:
        # Convert PDF to images
        images = convert_from_bytes(file_content.getvalue())

        # Convert images to base64 and create messages
        image_messages = [{
            "type": "text",
            "text": f"This is PDF named {file_name} has {len(images)} pages"
        }]
        for image in images:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_messages.append({
                "type": "text",
                "text": f"This is page {images.index(image) + 1} of PDF named {file_name}"
            })
            image_messages.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_base64,
                },
            })
        image_messages.append({
            "type": "text",
            "text": "End of PDF"
        })
        return image_messages

    except Exception as e:
        logging.error(f"Error processing PDF: {str(e)} {traceback.format_exc()}")
        return None
