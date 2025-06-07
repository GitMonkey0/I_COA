import re

def _process_input(input_str: str) -> dict:
    """
    Process input string to extract text and images.
    Supports image tags like <image>path/to/image.jpg</image> in the text.

    :param input_str: Input string potentially containing image tags
    :return: Dictionary with processed content
    """
    # Find all image tags in the input
    image_tags = re.findall(r'<image>(.*?)</image>', input_str)
    images = [img_path for img_path in image_tags]

    # Remove image tags to get clean text
    text = re.sub(r'<image>.*?</image>', '', input_str).strip()

    return {
        "text": text,
        "images": images
    }

print(_process_input("<image>a/1.jpg</image>"))