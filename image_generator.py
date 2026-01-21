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


def create_images(hebrew_prompt, picture_type, image_output_location, text_output_location, braille_output_location):
    # font_path = "/content/NotoSansSymbols2-Regular.ttf"
    font_path = "NotoSansSymbols2-Regular.ttf"

    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Noto Sans Symbols2'

    
    # general settings of Stable Diffusion
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device", device)

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
    
    general_prompt = lf.hebrew_translator(hebrew_prompt)
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
        
    #plotting final result
    # --- 1. Image only ---
    plt.figure(figsize=(5,5))
    plt.imshow(centered, cmap="gray")
    plt.axis("off")
    plt.savefig(image_output_location, dpi=300, bbox_inches="tight", pad_inches=0)#"image_only.png"
    plt.show()
    plt.close()

    # --- 2. Hebrew text only ---
    plt.figure(figsize=(5,5))
    plt.gca().set_facecolor("white")
    plt.text(0.5, 0.9, f'{hebrew_prompt[::-1]}',fontsize=30, color='black', ha='center', va='center', fontweight='light',fontname='DejaVu Sans')
    plt.axis("off")
    plt.savefig(text_output_location, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.show()
    plt.close()

    # --- 3. Braille text only ---
    plt.figure(figsize=(5,5))
    plt.gca().set_facecolor("white")
    plt.text(0.5, 0.1, f'{Braille}',fontsize=30, color='black', ha='center', va='center', fontweight='light',fontname='Noto Sans Symbols2')
    plt.axis("off")
    plt.savefig(braille_output_location, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.show()
    plt.close()

def images_to_dxf(image_location, text_location, braille_location):
    # Saving as dxf
    dxf_image_location, dxf_text_location, dxf_braille_location = image_location.replace('.png','dxf'), text_location.replace('.png','dxf'), braille_location.replace('.png','dxf')
    imf.image_to_dxf_exact(image_location, dxf_image_location)
    imf.png_to_dxf(text_location, dxf_text_location)
    imf.png_to_dxf(braille_location, dxf_braille_location)
    return dxf_image_location, dxf_text_location, dxf_braille_location

    # Visualize the results
    # imf.plot_dxf("image_only.dxf")
    # imf.plot_dxf("hebrew.dxf")
    # imf.plot_dxf("braille.dxf")