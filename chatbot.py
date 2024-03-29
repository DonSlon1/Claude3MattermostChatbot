import base64
import concurrent.futures
import datetime
import json
import logging
import os
import re
import ssl
import threading
import time
import traceback
from io import BytesIO
from urllib.parse import urlparse
import certifi
import httpx
from PIL import Image
from anthropic import Anthropic
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from mattermostdriver.driver import Driver
from pdf_processing import process_pdf
import re
import subprocess

# loading variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# save the original create_default_context function so we can call it later
create_default_context_orig = ssl.create_default_context


# define a new create_default_context function that sets purpose to ssl.Purpose.SERVER_AUTH
def cdc(*args, **kwargs):
    kwargs["cafile"] = certifi.where()  # Use certifi's CA bundle
    kwargs["purpose"] = ssl.Purpose.SERVER_AUTH
    return create_default_context_orig(*args, **kwargs)


# monkey patching ssl.create_default_context to fix SSL error
ssl.create_default_context = cdc

# AI parameters
api_key = os.environ["AI_API_KEY"]
model = os.getenv("AI_MODEL", "claude-3-opus-20240229")
timeout = int(os.getenv("AI_TIMEOUT", "120"))
max_tokens = int(os.getenv("MAX_TOKENS", "4096"))
temperature = float(os.getenv("TEMPERATURE", "0.15"))

# Mattermost server details
mattermost_url = os.environ["MATTERMOST_URL"]
mattermost_personal_access_token = os.getenv("MATTERMOST_TOKEN", "")
mattermost_ignore_sender_id = os.getenv("MATTERMOST_IGNORE_SENDER_ID", "")
mattermost_username = os.getenv("MATTERMOST_USERNAME", "")
mattermost_password = os.getenv("MATTERMOST_PASSWORD", "")
mattermost_mfa_token = os.getenv("MATTERMOST_MFA_TOKEN", "")

# Maximum website size
max_response_size = 1024 * 1024 * int(os.getenv("MAX_RESPONSE_SIZE_MB", "100"))

# For filtering local links
regex_local_links = r"(?:127\.|192\.168\.|10\.|172\.1[6-9]\.|172\.2[0-9]\.|172\.3[0-1]\.|::1|[fF][cCdD]|localhost)"

# Create a driver instance
driver = Driver(
    {
        "url": mattermost_url,
        "token": mattermost_personal_access_token,
        "login_id": mattermost_username,
        "password": mattermost_password,
        "mfa_token": mattermost_mfa_token,
        "scheme": "https",
        "port": 443,
        "basepath": "/api/v4",
        "verify": True,
    }
)

# Chatbot account username, automatically fetched
chatbot_username = ""
chatbot_usernameAt = ""

# Create an AI client instance
ai_client = Anthropic(api_key=api_key)

# Create a thread pool with a fixed number of worker threads
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)


def get_channel_history(channel_id, limit=50):
    try:
        # Fetch the latest 'limit' posts from the channel
        posts = driver.posts.get_posts_for_channel(channel_id, params={'per_page': limit, 'page': 0})
        return sorted(posts['posts'].values(), key=lambda post: post['create_at'])
    except Exception as e:
        logging.error(f"Error retrieving channel history: {str(e)} {traceback.format_exc()}")
        return []


def limit_images(messages, image_limit=20):
    image_count = 0
    limited_messages = []

    # Loop through each message and its content
    for message in messages:
        new_content = []
        for content in message['content']:
            # If content is an image, increment the count and check the limit
            if content['type'] == 'image':
                if image_count >= image_limit:
                    continue  # Skip if the image limit is reached
                image_count += 1

            # Add text or image to the new content, only if limit not hit
            new_content.append(content)

        # Replace old content with limited content
        message['content'] = new_content
        limited_messages.append(message)

    return limited_messages


def get_system_instructions():
    current_time = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[
                   :-3
                   ]
    return (
        f"You are a helpful assistant. The current UTC time is {current_time}. Whenever users ask you for help, "
        f"you will provide them with succinct answers formatted using Markdown; do not unnecessarily greet people "
        f"with their name. Do not be apologetic. You know the user's name as it is provided within [CONTEXT, "
        f"from:username] bracket at the beginning of a user-role message. Never add any CONTEXT bracket to your "
        f"replies (eg. [CONTEXT, from:{chatbot_username}]). The CONTEXT bracket may also include grabbed text "
        f"from a website if a user adds a link to their question.\n\n"
        f"When a user sends a PDF file, the file will be processed and sent to you as a series of images, each "
        f"representing a page of the PDF. The images will be accompanied by text messages indicating the name of "
        f"the PDF file, the total number of pages, and the page number of each image. When responding to questions "
        f"about a PDF, make sure to reference the specific pages and content of the PDF based on the provided images "
        f"and their associated page numbers. If a user asks about a PDF without providing the file, let them know "
        f"that you need the PDF file to be able to assist them effectively.\n\n"
        f"If you provide a Python code block in your response, the code will be executed automatically after your "
        f"response is sent. You can't include the expected output of the code in your response, as the actual "
        f"output will be appended to the message. Focus on providing clear and concise code without worrying about "
        f"the specific output."
    )


def sanitize_username(username):
    if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", username):
        username = re.sub(r"[.@!?]", "", username)[:64]
    if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", username):
        username = "".join(re.findall(r"[a-zA-Z0-9_-]", username))[:64]
    return username


def get_username_from_user_id(user_id):
    try:
        user = driver.users.get_user(user_id)
        return sanitize_username(user["username"])
    except Exception as e:
        logger.error(
            f"Error retrieving username for user ID {user_id}: {str(e)} {traceback.format_exc()}"
        )
        return f"Unknown_{user_id}"


def ensure_alternating_roles(messages):
    updated_messages = []
    for i in range(len(messages)):
        if i > 0 and messages[i]["role"] == messages[i - 1]["role"]:
            updated_messages.append(
                {
                    "role": "assistant" if messages[i]["role"] == "user" else "user",
                    "content": [
                        {"type": "text", "text": "[CONTEXT, acknowledged post]"}
                    ],
                }
            )
        updated_messages.append(messages[i])
    return updated_messages


def send_typing_indicator(user_id, channel_id, parent_id):
    """Send a "typing" indicator to show that work is in progress."""
    options = {
        "channel_id": channel_id,
        # "parent_id": parent_id  # somehow bugged/doesnt work
    }
    driver.client.make_request("post", f"/users/{user_id}/typing", options=options)


def send_typing_indicator_loop(user_id, channel_id, parent_id, stop_event):
    while not stop_event.is_set():
        try:
            send_typing_indicator(user_id, channel_id, parent_id)
            time.sleep(2)
        except Exception as e:
            logging.error(
                f"Error sending busy indicator: {str(e)} {traceback.format_exc()}"
            )


def handle_typing_indicator(user_id, channel_id, parent_id):
    stop_typing_event = threading.Event()
    typing_indicator_thread = threading.Thread(
        target=send_typing_indicator_loop,
        args=(user_id, channel_id, parent_id, stop_typing_event),
    )
    typing_indicator_thread.start()
    return stop_typing_event, typing_indicator_thread


def split_message(msg, max_length=4000):
    """
    Split a message based on a maximum character length, ensuring Markdown code blocks
    and their languages are preserved. Avoids unnecessary linebreaks at the end of the last message
    and when closing a code block if the last line is already a newline.

    Args:
    - msg (str): The message to be split.
    - max_length (int, optional): The maximum length of each split message. Defaults to 4000.

    Returns:
    - list of str: The message split into chunks, preserving Markdown code blocks and languages,
                   and avoiding unnecessary linebreaks.
    """
    if len(msg) <= max_length:
        return [msg]

    if len(msg) > 40000:
        raise Exception(f"Message too long.")

    current_chunk = ""  # Holds the current message chunk
    chunks = []  # Collects all message chunks
    in_code_block = False  # Tracks whether the current line is inside a code block
    code_block_lang = ""  # Keeps the language of the current code block

    # Helper function to add a chunk to the list
    def add_chunk(chunk, in_code, code_lang):
        if in_code:
            # Check if the last line is not just a newline itself
            if not chunk.endswith("\n\n"):
                chunk += "\n"
            chunk += "```"
        chunks.append(chunk)
        if in_code:
            # Start a new code block with the same language
            return f"```{code_lang}\n"
        return ""

    lines = msg.split('\n')
    for i, line in enumerate(lines):
        # Check if this line starts or ends a code block
        if line.startswith("```"):
            if in_code_block:
                # Ending a code block
                in_code_block = False
                # Avoid adding an extra newline if the line is empty
                if current_chunk.endswith("\n"):
                    current_chunk += line
                else:
                    current_chunk += "\n" + line
            else:
                # Starting a new code block, capture the language
                in_code_block = True
                code_block_lang = line[3:].strip()  # Remove the backticks and get the language
                current_chunk += line + "\n"
        else:
            # If adding this line exceeds the max length, we need to split here
            if len(current_chunk) + len(line) + 1 > max_length:
                # Split here, preserve the code block state and language if necessary
                current_chunk = add_chunk(current_chunk, in_code_block, code_block_lang)
                current_chunk += line
                if i < len(lines) - 1:  # Avoid adding a newline at the end of the last line
                    current_chunk += "\n"
            else:
                current_chunk += line
                if i < len(lines) - 1:  # Avoid adding a newline at the end of the last line
                    current_chunk += "\n"

    # Don't forget to add the last chunk
    if current_chunk:
        add_chunk(current_chunk, in_code_block, code_block_lang)

    return chunks


def handle_text_generation(
        last_message, messages, channel_id, root_id, sender_name, links
):
    pdf_files_provided = any(
        "This is PDF named" in msg_text
        for msg in messages
        for content in msg.get("content", [])
        for key, msg_text in content.items() if key == 'text'
    )
    # Modify the system instructions based on whether PDF files were provided or not
    if pdf_files_provided:
        system_instructions = (
            f"{get_system_instructions()}\n\n"
            f"PDF files have been provided in this conversation. When responding, make sure to reference the "
            f"specific pages and content of the PDFs based on the images and their associated page numbers."
        )
    else:
        system_instructions = (
            f"{get_system_instructions()}\n\n"
            f"No PDF files have been provided in this conversation. If the user asks about a PDF, let them know "
            f"that you need the PDF file to be able to assist them effectively."
        )
    # Send the messages to the AI API
    response = ai_client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_instructions,
        messages=messages,
        timeout=timeout,
        temperature=temperature,
    )

    # Extract the text content from the ContentBlock object
    content_blocks = response.content
    response_text = ""
    for block in content_blocks:
        if block.type == "text":
            response_text += block.text

    # Check if the response contains a Python code block

    python_code_block = re.findall(r"```python\n(.*?)\n```", response_text, re.DOTALL)
    if python_code_block:

        # Execute the Python code and get the output
        code_output = execute_python_code(python_code_block[0])

        # Append the code output to the response text
        response_text += f"\n\nOutput of the Python script:\n{code_output}"

    # Failsafe: Remove all blocks containing [CONTEXT
    response_text = re.sub(r"(?s)\[CONTEXT.*?]", "", response_text).strip()

    # Failsafe: Remove all input links from the response
    for link in links:
        response_text = response_text.replace(link, "")
    response_text = response_text.strip()

    # Split the response into multiple messages if necessary
    response_parts = split_message(response_text)

    # Send each part of the response as a separate message
    for part in response_parts:
        # Send the API response back to the Mattermost channel as a reply to the thread or as a new thread
        driver.posts.create_post(
            {"channel_id": channel_id, "message": part, "root_id": root_id}
        )


def execute_python_code(code):
    try:
        # Write the code to a temporary file
        with open("temp_script.py", "w") as file:
            file.write(code)

        # Execute the Python script in a subprocess
        result = subprocess.run(["python", "temp_script.py"], capture_output=True, text=True)

        # Check if the script executed successfully
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error: {result.stderr}"

    except Exception as e:
        return f"Error: {str(e)}"

    finally:
        # Clean up the temporary file
        os.remove("temp_script.py")


def process_message(last_message, messages, channel_id, root_id, sender_name, links):
    stop_typing_event = None
    typing_indicator_thread = None
    try:
        logger.info("Querying AI API")

        # Start the typing indicator
        stop_typing_event, typing_indicator_thread = handle_typing_indicator(
            driver.client.userid, channel_id, root_id
        )

        handle_text_generation(
            last_message, messages, channel_id, root_id, sender_name, links
        )

    except Exception as e:
        logging.error(f"Error: {str(e)} {traceback.format_exc()}")
        driver.posts.create_post(
            {"channel_id": channel_id, "message": str(e), "root_id": root_id}
        )

    finally:
        if stop_typing_event is not None:
            stop_typing_event.set()
        if typing_indicator_thread is not None:
            typing_indicator_thread.join()


async def message_handler(event):
    try:
        event_data = json.loads(event)
        logging.info(f"Received event: {event_data}")
        if event_data.get("event") == "hello":
            logging.info("Received 'hello' event. WebSocket connection established.")
        elif event_data.get("event") == "posted":

            post = json.loads(event_data["data"]["post"])
            sender_id = post["user_id"]
            if (
                    sender_id == driver.client.userid
                    or sender_id == mattermost_ignore_sender_id
            ):
                logging.info("Ignoring post from a ignored sender ID")
                return

            # Check if the post is from a bot
            if post.get("props", {}).get("from_bot") == "true":
                logging.info("Ignoring post from a bot")
                return

            # Remove the "@chatbot" mention from the message
            message = post["message"].replace(chatbot_usernameAt, "").strip()
            channel_id = post["channel_id"]
            sender_name = sanitize_username(event_data["data"]["sender_name"])
            root_id = post["root_id"]  # Get the root_id of the thread
            post_id = post["id"]
            channel_display_name = event_data["data"]["channel_display_name"]

            try:
                # Define the messages list outside the if-else
                messages = []
                thread_files = []

                # Retrieve the channel context
                channel_id = post["channel_id"]
                channel_posts = get_channel_history(channel_id)  # Change limit as needed
                for channel_post in channel_posts:
                    sender_name = sanitize_username(get_username_from_user_id(channel_post["user_id"]))
                    message_content = channel_post["message"]
                    # Check for files in each post
                    file_messages = get_message_files(channel_post)  # Using the previously defined function
                    role = "assistant" if channel_post["user_id"] == driver.client.userid else "user"

                    # Compose the message object
                    message_obj = {
                        "role": role,
                        "content": [
                            {
                                "type": "text",
                                "text": f"[CONTEXT, from:{sender_name}] {message_content}",
                            }
                        ],
                    }

                    # Append file messages if there are any
                    if file_messages:
                        message_obj["content"].extend(file_messages)

                    messages.append(message_obj)
                # Retrieve the thread context
                chatbot_invoked = False
                if root_id:
                    thread = driver.posts.get_thread(root_id)
                    # Sort the thread posts based on their create_at timestamp
                    sorted_posts = sorted(
                        thread["posts"].values(), key=lambda x: x["create_at"]
                    )
                    for thread_post in sorted_posts:
                        if thread_post["id"] != post_id:
                            thread_sender_name = get_username_from_user_id(
                                thread_post["user_id"]
                            )
                            thread_message = thread_post["message"]
                            role = (
                                "assistant"
                                if thread_post["user_id"] == driver.client.userid
                                else "user"
                            )

                            # Add message files to the history
                            thread_files.extend(get_message_files(thread_post))

                            messages.append(
                                {
                                    "role": role,
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": f"[CONTEXT, from:{thread_sender_name}] {thread_message}",
                                        }
                                    ],
                                }
                            )

                            if role == "assistant":
                                chatbot_invoked = True
                else:
                    # If the message is not part of a thread, reply to it to create a new thread
                    root_id = post["id"]

                if thread_files:
                    messages.append({"role": "user", "content": thread_files})

                # Add the current message to the messages array if "@chatbot" is mentioned, the chatbot has already
                # been invoked in the thread or it's a DM
                if (
                        chatbot_usernameAt in post["message"]
                        or chatbot_invoked
                        or channel_display_name.startswith("@")
                ):
                    if "file_ids" in post:
                        file_ids = post["file_ids"]
                        logger.info(f"Found file_ids: {file_ids}")
                        file_messages = []
                        for file_id in file_ids:
                            try:
                                file_info = driver.files.get_file(file_id)
                                file_name = driver.files.get_file_metadata(file_id)["name"]
                                logger.info(f"Processing file: {file_info.headers['Content-Disposition']}")

                                if file_info.headers['Content-Type'] == 'application/pdf':
                                    pdf_messages = process_pdf(BytesIO(file_info.content), file_name)
                                    if pdf_messages:
                                        file_messages.extend(pdf_messages)
                                elif file_info.headers['Content-Type'].startswith('image/'):
                                    image_base64 = base64.b64encode(file_info.content).decode("utf-8")
                                    file_messages.append({
                                        "type": "text",
                                        "text": f"This is image have name of {file_name} and the type of this image is"
                                                f"{file_info.headers['Content-Type']}"
                                    })
                                    file_messages.append({
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": file_info.headers['Content-Type'],
                                            "data": image_base64,
                                        },
                                    })
                                else:
                                    logger.info(f"Unsupported file type: {file_info.headers['Content-Type']}")

                            except Exception as e:
                                logging.error(f"Error processing file: {str(e)} {traceback.format_exc()}")

                        if file_messages:
                            messages.append({"role": "user", "content": file_messages})
                            logger.info(f"Sending file messages to AI: {file_messages}")
                    else:
                        logger.info("No file_ids found in the post")
                    links = re.findall(
                        r"(https?://\S+)", message
                    )  # Allow both http and https links
                    # add images to links

                    extracted_text = ""
                    total_size = 0
                    image_messages = []
                    file_messages = []

                    with httpx.Client() as client:
                        for link in links:
                            if re.search(regex_local_links, link):
                                logging.info(f"Skipping local URL: {link}")
                                continue
                            try:
                                with client.stream("GET", link, timeout=4, follow_redirects=True) as response:
                                    final_url = str(response.url)

                                    if re.search(regex_local_links, final_url):
                                        logging.info(f"Skipping local URL after redirection: {final_url}")
                                        continue

                                    content_type = response.headers.get("content-type", "").lower()
                                    content_disposition = response.headers.get("content-disposition")
                                    file_name = None

                                    if content_disposition:
                                        # Pokus o získání názvu souboru z hlavičky Content-Disposition
                                        cd_params = content_disposition.split(";")
                                        for param in cd_params:
                                            param = param.strip()
                                            if param.startswith("filename="):
                                                file_name = param[9:].strip('"')
                                                break

                                    if not file_name:
                                        # Pokud název souboru není v hlavičce Content-Disposition, získejte ho z URL
                                        file_name = os.path.basename(urlparse(final_url).path)
                                    if content_type == "application/pdf":
                                        try:
                                            pdf_content = BytesIO()
                                            for chunk in response.iter_bytes():
                                                pdf_content.write(chunk)
                                                total_size += len(chunk)
                                                if total_size > max_response_size:
                                                    extracted_text += ("*WEBSITE SIZE EXCEEDED THE MAXIMUM LIMIT FOR "
                                                                       "THE CHATBOT, WARN THE CHATBOT USER*")
                                                    raise Exception("Response size exceeds the maximum limit")
                                            pdf_messages = process_pdf(pdf_content, file_name)
                                            if pdf_messages:
                                                file_messages.extend(pdf_messages)
                                        except Exception as e:
                                            logging.error(
                                                f"Error processing PDF from link {link}: {str(e)} {traceback.format_exc()}")
                                            file_messages.append({"type": "text",
                                                                  "text": f"The provided link {link} does not lead to "
                                                                          f"a valid PDF file."})

                                    elif "image" in content_type:
                                        # Check for compatible content types
                                        compatible_content_types = [
                                            "image/jpeg",
                                            "image/png",
                                            "image/gif",
                                            "image/webp",
                                        ]
                                        if content_type not in compatible_content_types:
                                            raise Exception(
                                                f"Unsupported image content type: {content_type}"
                                            )

                                        # Handle image content
                                        image_data = b""
                                        for chunk in response.iter_bytes():
                                            image_data += chunk
                                            total_size += len(chunk)
                                            if total_size > max_response_size:
                                                extracted_text += ("*WEBSITE SIZE EXCEEDED THE MAXIMUM LIMIT FOR THE "
                                                                   "CHATBOT, WARN THE CHATBOT USER*")
                                                raise Exception(
                                                    "Response size exceeds the maximum limit at image processing"
                                                )

                                        # Open the image using Pillow
                                        image = Image.open(BytesIO(image_data))

                                        # Calculate the aspect ratio of the image
                                        width, height = image.size
                                        aspect_ratio = width / height

                                        # Define the supported aspect ratios and their corresponding dimensions
                                        supported_ratios = [
                                            (1, 1, 1092, 1092),
                                            (0.75, 3 / 4, 951, 1268),
                                            (0.67, 2 / 3, 896, 1344),
                                            (0.56, 9 / 16, 819, 1456),
                                            (0.5, 1 / 2, 784, 1568),
                                        ]

                                        # Find the closest supported aspect ratio
                                        closest_ratio = min(
                                            supported_ratios,
                                            key=lambda x: abs(x[0] - aspect_ratio),
                                        )
                                        target_width, target_height = (
                                            closest_ratio[2],
                                            closest_ratio[3],
                                        )

                                        # Resize the image to the target dimensions
                                        resized_image = image.resize(
                                            (target_width, target_height),
                                            Image.Resampling.LANCZOS,
                                        )

                                        # Save the resized image to a BytesIO object
                                        buffer = BytesIO()
                                        resized_image.save(
                                            buffer, format=image.format, optimize=True
                                        )
                                        resized_image_data = buffer.getvalue()

                                        # Check if the resized image size exceeds 3MB
                                        if len(resized_image_data) > 3 * 1024 * 1024:
                                            # Compress the resized image to a target size of 3MB
                                            target_size = 3 * 1024 * 1024
                                            quality = 90
                                            while len(resized_image_data) > target_size:
                                                # Reduce the image quality until the size is within the target
                                                buffer = BytesIO()
                                                resized_image.save(
                                                    buffer,
                                                    format=image.format,
                                                    optimize=True,
                                                    quality=quality,
                                                )
                                                resized_image_data = buffer.getvalue()
                                                quality -= 5

                                                if quality <= 0:
                                                    raise Exception(
                                                        "Image too large, can't compress"
                                                    )

                                        image_data_base64 = base64.b64encode(
                                            resized_image_data
                                        ).decode("utf-8")
                                        image_messages.append({
                                            "type": "text",
                                            "text": f"This is image have name of {file_name} and the type of this image is"
                                                    f"{content_type}"
                                        })
                                        image_messages.append(
                                            {
                                                "type": "image",
                                                "source": {
                                                    "type": "base64",
                                                    "media_type": content_type,
                                                    "data": image_data_base64,
                                                },
                                            }
                                        )
                                    else:
                                        # Handle text content
                                        content_chunks = []
                                        for chunk in response.iter_bytes():
                                            content_chunks.append(chunk)
                                            total_size += len(chunk)
                                            if total_size > max_response_size:
                                                extracted_text += ("*WEBSITE SIZE EXCEEDED THE MAXIMUM LIMIT FOR THE "
                                                                   "CHATBOT, WARN THE CHATBOT USER*")
                                                raise Exception(
                                                    "Response size exceeds the maximum limit"
                                                )
                                        content = b"".join(content_chunks)
                                        soup = BeautifulSoup(content, "html.parser")
                                        extracted_text += soup.get_text()
                            except Exception as e:
                                logging.error(
                                    f"Error extracting content from link {link}: {str(e)} {traceback.format_exc()}"
                                )

                    content = f"[CONTEXT, from:{sender_name}"
                    if extracted_text != "":
                        content += f", extracted_website_text:{extracted_text}"
                    content += f"] {message}"

                    all_messages = image_messages + file_messages
                    if all_messages:
                        all_messages.append({"type": "text", "text": content})
                        messages.append({"role": "user", "content": all_messages})
                    else:
                        messages.append(
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": content}],
                            }
                        )

                    # Ensure alternating roles in the messages array
                    messages = ensure_alternating_roles(messages)
                    messages = limit_images(messages)

                    # Submit the task to the thread pool. We do this because Mattermostdriver-async is outdated
                    thread_pool.submit(
                        process_message,
                        message,
                        messages,
                        channel_id,
                        root_id,
                        sender_name,
                        links,
                    )

            except Exception as e:
                logging.error(
                    f"Error inner message handler: {str(e)} {traceback.format_exc()}"
                )
        else:
            # Handle other events
            pass
    except json.JSONDecodeError:
        logging.error(
            f"Failed to parse event as JSON: {event} {traceback.format_exc()}"
        )
    except Exception as e:
        logging.error(f"Error message_handler: {str(e)} {traceback.format_exc()}")


def main():
    try:
        # Log in to the Mattermost server
        driver.login()
        global chatbot_username, chatbot_usernameAt
        chatbot_username = driver.client.username
        chatbot_usernameAt = f"@{chatbot_username}"

        # Initialize the WebSocket connection
        while True:
            try:
                # Initialize the WebSocket connection
                driver.init_websocket(message_handler)
            except Exception as e:
                logging.error(
                    f"Error initializing WebSocket: {str(e)} {traceback.format_exc()}"
                )
            time.sleep(2)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt, logout and exit")
        driver.logout()
    except Exception as e:
        logging.error(f"Error: {str(e)} {traceback.format_exc()}")


def get_message_files(thread_post):
    file_messages = []
    if "file_ids" in thread_post and thread_post["file_ids"]:
        file_ids = thread_post["file_ids"]
        for file_id in file_ids:
            file_info = driver.files.get_file(file_id)
            file_name = driver.files.get_file_metadata(file_id)["name"]
            content_type = file_info.headers['Content-Type']
            if content_type.startswith('image/'):
                image_base64 = base64.b64encode(file_info.content).decode("utf-8")
                file_messages.append({
                    "type": "text",
                    "text": f"tis image have name of {file_name} and the type of this image is {content_type}"
                })
                file_messages.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": content_type,
                        "data": image_base64,
                    },
                })
            elif content_type == 'application/pdf':
                pdf_messages = process_pdf(BytesIO(file_info.content), file_name)
                if pdf_messages:
                    file_messages.extend(pdf_messages)
    return file_messages


if __name__ == "__main__":
    main()
