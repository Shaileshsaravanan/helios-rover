from rembg import remove
from PIL import Image
import io
import base64

def remove_background_from_base64(base64_str):
    image_data = base64.b64decode(base64_str)
    output_image_data = remove(image_data)
    output_image_pil = Image.open(io.BytesIO(output_image_data))
    output_image_pil.show()
    buffered = io.BytesIO()
    output_image_pil.save(buffered, format="PNG")
    new_base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return new_base64_str

example_base64_str = ""

new_base64_str = remove_background_from_base64(example_base64_str)
print(new_base64_str) 