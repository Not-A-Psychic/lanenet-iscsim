import os
import random
from glob import glob

IMAGE_DIR = '/media/jai/Deck/projects/ISC/lanenet/dataset/iscwebots/images'
OUTPUT_DIR = '/media/jai/Deck/projects/ISC/lanenet/dataset/iscwebots/trainer'

all_images = sorted(glob(os.path.join(IMAGE_DIR, '*.png')))
random.shuffle(all_images)

split_ratio = 0.8
split_idx = int(len(all_images) * split_ratio)

train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

with open(os.path.join(OUTPUT_DIR, 'train.txt'), 'w') as f:
    for img in train_images:
        f.write('images/' + os.path.basename(img) + '\n')

with open(os.path.join(OUTPUT_DIR, 'val.txt'), 'w') as f:
    for img in val_images:
        f.write('images/' + os.path.basename(img) + '\n')

print(f"Train: {len(train_images)} images")
print(f"Val:   {len(val_images)} images")

