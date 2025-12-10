
import cv2
import ezdxf
import numpy as np

def img_to_dxf(img_location, dxf_location="output.dxf"):
    # Load image
    img = cv2.imread(img_location, cv2.IMREAD_GRAYSCALE)

    # Threshold to black/white
    _, th = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create DXF
    doc = ezdxf.new()
    msp = doc.modelspace()

    for cnt in contours:
        # Convert contour to (x, y) points
        points = [(float(p[0][0]), float(p[0][1])) for p in cnt]
        # Add polyline to DXF
        msp.add_lwpolyline(points, close=True)

    doc.saveas(dxf_location)