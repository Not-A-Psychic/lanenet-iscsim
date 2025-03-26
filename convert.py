import os
import cv2
import numpy as np
from glob import glob

IMAGE_DIR = '/media/jai/Deck/projects/ISC/lanenet/findat/data/images'
LABEL_DIR = '/media/jai/Deck/projects/ISC/lanenet/findat/labels/train'
INSTANCE_DIR = './gt_instance_image'
BINARY_DIR = './gt_binary_image'

os.makedirs(INSTANCE_DIR, exist_ok=True)
os.makedirs(BINARY_DIR, exist_ok=True)

def get_image_shape(image_path):
    img = cv2.imread(image_path)
    return img.shape[:2]  # (height, width)

def denormalize_points(coords, width, height):
    pts = []
    for i in range(0, len(coords), 2):
        x = int(float(coords[i]) * width)
        y = int(float(coords[i+1]) * height)
        pts.append([x, y])
    return np.array([pts], dtype=np.int32)

for label_path in glob(os.path.join(LABEL_DIR, '*.txt')):
    base = os.path.basename(label_path).replace('.txt', '')
    image_path = os.path.join(IMAGE_DIR, base + '.png')  # or .jpg
    if not os.path.exists(image_path):
        print(f"Image not found: {base}")
        continue

    h, w = get_image_shape(image_path)
    instance_mask = np.zeros((h, w), dtype=np.uint8)
    binary_mask = np.zeros((h, w), dtype=np.uint8)

    with open(label_path, 'r') as f:
        lines = f.readlines()

    instance_id = 1
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue

        class_id = int(parts[0])

        # Only include: dotted_lines, solid_line, finish, obstacle
        if class_id not in [0, 1, 2, 4]:
            continue

        polygon = denormalize_points(parts[1:], w, h)
        if polygon.size == 0:
            continue

        cv2.fillPoly(instance_mask, polygon, color=instance_id)
        cv2.fillPoly(binary_mask, polygon, color=255)

        label_map = {0: 'dotted_lines', 1: 'solid_line', 2: 'finish', 4: 'obstacle'}
        print(f"Added {label_map.get(class_id, 'unknown')} (class {class_id}) as instance {instance_id}")

        instance_id += 1


    cv2.imwrite(os.path.join(INSTANCE_DIR, base + '.png'), instance_mask)
    cv2.imwrite(os.path.join(BINARY_DIR, base + '.png'), binary_mask)
