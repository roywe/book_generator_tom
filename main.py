# colab installs
# !pip install deep_translator
# !wget -O NotoSansSymbols2-Regular.ttf https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansSymbols2/NotoSansSymbols2-Regular.ttf
# !pip install ezdxf

# Imports
import torch
from diffusers import AutoPipelineForText2Image
import cv2
import numpy as np
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import language_funcs as lf
import image_funcs as imf

# font_path = "/content/NotoSansSymbols2-Regular.ttf"
font_path = "NotoSansSymbols2-Regular.ttf"

fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Noto Sans Symbols2'

# device to be used
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Using device", device)

# general settings of Stable Diffusion
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    # GPU → float16
    pipe = AutoPipelineForText2Image.from_pretrained(
        "segmind/SSD-1B",
        torch_dtype=torch.float16
    ).to(device)

else:
    # CPU → float32
    pipe = AutoPipelineForText2Image.from_pretrained(
        "segmind/SSD-1B",
        torch_dtype=torch.float32
    ).to(device)

## text to image generator from user's input ##

# desired text prompt
hebrew_prompt  = input(" :בבקשה הכנס/י את התמונה שתרצי/שתרצה בספר")
general_prompt = lf.hebrew_translator(hebrew_prompt)
picture_type  = input(" :בבקשה הכנס/י את הקטגוריה של התמונה (פרי, ירק, מספר, אות)")
picture_type = lf.hebrew_translator(picture_type)
user_prompt = (
    f"A single vectorized clean line-art illustration of {general_prompt}, "
    f"classified as a {picture_type}. "
    "Minimalistic outline, high contrast white background, no shading, no texture, "
    "sharp edges, centered composition, no extra objects, no text, no noise."
)
user_prompt = [user_prompt]

# Generate the image
image = pipe(user_prompt, num_inference_steps=25, guidance_scale=7.5).images[0]

# plot results
plt.imshow(image, cmap="gray")
plt.axis("off")
plt.show()

## Braille converter
hebrew_prompt = lf.add_nikud(hebrew_prompt)
Braille = lf.convert_to_braille(hebrew_prompt)
print(Braille)

#image processing
image = np.array(image)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray_image, 50, 200)
kernel = np.ones((5, 5), np.uint8)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
edges = cv2.bitwise_not(edges)
h, w = edges.shape
edges[h-1:h, w-1:w] = 255

#centering object
h, w = edges.shape
ys, xs = np.where(edges[1:h-1,1:w-1] == 0)
obj_cx = xs.mean()
obj_cy = ys.mean()
img_cx = w / 2
img_cy = h / 2
shift_x = int(img_cx - obj_cx)
shift_y = int(img_cy - obj_cy)
M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
centered = cv2.warpAffine(edges, M, (w, h), borderValue=255)

# Saving as dxf
imf.image_to_dxf_exact(centered, "image_only.dxf")
imf.png_to_dxf("hebrew.png", "hebrew.dxf")
imf.png_to_dxf("braille.png", "braille.dxf")

# Visualize the results
imf.plot_dxf("image_only.dxf")
imf.plot_dxf("hebrew.dxf")
imf.plot_dxf("braille.dxf")