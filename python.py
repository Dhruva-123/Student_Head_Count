# Headcount Model Project - Batch People Counting Script
# Run: python count_people_batch.py --folder path/to/images

from ultralytics import YOLO
import os
import argparse
import csv

MODEL_PATH = "yolo11n.pt"  # Pretrained YOLO model

# Load YOLO model
model = YOLO(MODEL_PATH)


def count_people(image_path):
    results = model(image_path)[0]
    count = sum(1 for det in results.boxes if int(det.cls) == 0)
    return count


def main(folder_path, output_csv):
    if not os.path.exists(folder_path):
        print("Folder not found.")
        return

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("No images found in folder.")
        return

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image', 'People_Count'])
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            count = count_people(img_path)
            writer.writerow([img_file, count])
            print(f"{img_file}: {count} people")

    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help=r"D:\smart_tech\archive\ucsdpeds\vidf\vidf2_33_009.y")
    parser.add_argument("--output", default="people_counts.csv", help="CSV output file")
    args = parser.parse_args()

    main(args.folder, args.output)
