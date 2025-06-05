from torchvision import transforms
import os
from PIL import Image
import csv

MAPPING_CSV = "dataset/annotations/image_annotations_2.csv"

augment_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomRotation(15),
])

def main():
    phase_23_dir = "dataset/images/train/phase_23"
    list_of_images = sorted([image.split(".")[0] for image in os.listdir(phase_23_dir)])
    target_augmentation_count = 500 - 366
    start_index = list_of_images.index("case_0731_006643")

    # First, read all rows from the CSV file
    all_rows = []
    with open(MAPPING_CSV, "r") as f:
        reader = csv.reader(f)
        all_rows = list(reader)

    # Process each image and update CSV data
    for i in range(target_augmentation_count):
        image_name = list_of_images[start_index + i % (len(list_of_images) - start_index)]
        image_path = os.path.join(phase_23_dir, f"{image_name}.png")
        image = Image.open(image_path)
        image = augment_transform(image)
        image.save(os.path.join(phase_23_dir, f"{image_name}_aug.png"))

        # Find the original image row and insert augmented image row after it
        for j, row in enumerate(all_rows):
            if row[2] == f"{image_name}.png":
                new_row = [row[0], row[1], f"{image_name}_aug.png", row[3], 23, True, row[6]]
                all_rows.insert(j + 1, new_row)
                break

    # Write all rows back to the CSV file
    with open(MAPPING_CSV, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(all_rows)

if __name__ == "__main__":
    main()