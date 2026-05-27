# Imports
import torch
import cv2
import numpy as np
import ezdxf
from PIL import Image

# functions
def convert_tensor_to_pil_img(tensor):
    """
    function converts a tensor of size CxHxW in [-1,1] to a PIL image in [0,255]
    :param x (torch.Tensor): input tensor.
    :return image (PIL.Image): output image.
    """
    image = (tensor / 2 + 0.5).clamp(0, 1).squeeze()
    image = (image.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
    image = Image.fromarray(image)
    return image

def image_to_dxf_exact(image_bw, out_path, canvas_cm=150):
    canvas_mm = canvas_cm * 10.0

    img = image_bw.copy()
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # Ensure white background, black lines
    if np.count_nonzero(img < 128) > np.count_nonzero(img >= 128):
        img = cv2.bitwise_not(img)

    # Extract thin edges ONLY (no morphology)
    edges = cv2.Canny(img, 50, 150)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        edges, connectivity=8
    )

    MIN_AREA = 80   # px — safe for thin lines, removes junk dots

    clean = np.zeros_like(edges)

    for i in range(1, num_labels):  # skip background
        if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA:
            clean[labels == i] = 255

    edges = clean

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        raise RuntimeError("No contours found")

    all_pts = np.vstack([c.reshape(-1, 2) for c in contours])
    min_x, min_y = all_pts.min(axis=0)
    max_x, max_y = all_pts.max(axis=0)

    w_px = max_x - min_x + 1
    h_px = max_y - min_y + 1

    scale = canvas_mm / max(w_px, h_px)
    offset_x = (canvas_mm - w_px * scale) / 2
    offset_y = (canvas_mm - h_px * scale) / 2

    def px_to_mm(p):
        x = (p[0] - min_x) * scale + offset_x
        y = (max_y - p[1]) * scale + offset_y
        return (x, y)

    doc = ezdxf.new(setup=True)
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()

    for c in contours:
        pts = [px_to_mm(p[0]) for p in c]
        if len(pts) > 1:
            msp.add_lwpolyline(pts, close=True)

    doc.saveas(out_path)

def png_to_dxf(png_path, dxf_path, canvas_cm=150):
    canvas_mm = canvas_cm * 10.0

    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not load {png_path}")

    # binarize
    _, bw = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        bw,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise RuntimeError("No contours found")

    all_pts = np.vstack([c.reshape(-1,2) for c in contours])
    min_x, min_y = all_pts.min(axis=0)
    max_x, max_y = all_pts.max(axis=0)

    w_px = max_x - min_x + 1
    h_px = max_y - min_y + 1

    scale = canvas_mm / max(w_px, h_px)
    offset_x = (canvas_mm - w_px * scale) / 2
    offset_y = (canvas_mm - h_px * scale) / 2

    def px_to_mm(p):
        x = (p[0] - min_x) * scale + offset_x
        y = (max_y - p[1]) * scale + offset_y
        return (x, y)

    doc = ezdxf.new(setup=True)
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()

    for c in contours:
        pts = [px_to_mm(p[0]) for p in c]
        if len(pts) > 1:
            msp.add_lwpolyline(pts, close=True)

    doc.saveas(dxf_path)

def plot_dxf(dxf_path):
    import ezdxf
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()

        plt.figure(figsize=(6, 6))

        for entity in msp:
            if entity.dxftype() == 'LWPOLYLINE':
                points = entity.get_points()
                x = [p[0] for p in points]
                y = [p[1] for p in points]

                if entity.is_closed:
                    x.append(x[0])
                    y.append(y[0])

                plt.plot(x, y, color='black', linewidth=1)

            elif entity.dxftype() == 'CIRCLE':
                center = entity.dxf.center
                radius = entity.dxf.radius

                theta = np.linspace(0, 2*np.pi, 100)
                x = center.x + radius * np.cos(theta)
                y = center.y + radius * np.sin(theta)

                plt.plot(x, y, color='black', linewidth=1)

        plt.axis('equal')
        plt.title(f"DXF Preview: {dxf_path}")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"Could not plot DXF: {e}")