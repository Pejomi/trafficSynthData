import os
import json
from PIL import Image
import shutil
import torch
from ultralytics import YOLO


def convert_annotations(data_root, output_label_dir):
    # Ensure the output directory exists
    os.makedirs(output_label_dir, exist_ok=True)

    for dirs in os.listdir(data_root):
        for file in os.listdir(os.path.join(data_root, dirs)):
            if file.endswith('.json'):
                json_path = os.path.join(data_root + "/" + dirs, file)
                image_filename = "step0.camera.png"
                image_path = os.path.join(data_root + "/" + dirs, image_filename)
                label_filename = dirs + ".txt"
                label_path = os.path.join(output_label_dir, label_filename)

                # Open image to get dimensions
                with Image.open(image_path) as img:
                    img_width, img_height = img.size

                # Load JSON file
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # Process annotations
                with open(label_path, 'w') as label_file:
                    for capture in data['captures']:
                        for annotation in capture['annotations']:
                            for value in annotation['values']:
                                x_min, y_min = value['origin']
                                width, height = value['dimension']

                                # Convert to YOLO format
                                x_center = (x_min + width / 2) / img_width
                                y_center = (y_min + height / 2) / img_height
                                norm_width = width / img_width
                                norm_height = height / img_height

                                # Write to label file
                                class_id = value['labelId'] - 1  # Assuming class_id needs to be zero-indexed
                                label_file.write(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}\n")


def separate_pngs(data_root, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for dirs in os.listdir(data_root):
        for file in os.listdir(os.path.join(data_root, dirs)):
            if file.endswith('.png'):
                print(f"Copying {file} to {output_dir}")
                shutil.copy(os.path.join(data_root + "/" + dirs, file), os.path.join(output_dir, dirs + ".png"))


def train():
    # torch.cuda.set_device(0)
    # device = torch.device("cuda")

    model = YOLO('yolov8n.pt')
    # model.to(device=device)

    # Start training
    model.train(
        data='dataset.yaml',  # Path to your dataset configuration file
        epochs=30  # Number of epochs to train for
    )

    # Evaluate the model's performance on the validation set
    results = model.val()

    print(results)


