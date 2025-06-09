import os
import csv
import matplotlib.pyplot as plt

def plot_phase_distribution(
    annotations_file: str,
    image_dir: str,
    out_dir: str,
    split: str = "train",
):
    """
    Reads annotations CSV, counts how many images per phase exist in image_dir/<split>/phase_{XX}/,
    plots a bar chart, saves it to out_dir/phase_images_distribution_<split>.png,
    and prints summary statistics.

    Expected CSV columns: image_name, phase, [split]
    Expected directory structure: image_dir/<split>/phase_00/, phase_01/, ..., phase_34/
    """

    num_classes = 35  
    image_counts = [0] * num_classes
    split_lower = split.strip().lower()

    phase_to_filenames = {}
    for phase_id in range(num_classes):
        phase_str = f"{phase_id:02d}"
        folder_path = os.path.join(image_dir, split_lower, f"phase_{phase_str}")

        if os.path.isdir(folder_path):
            phase_to_filenames[phase_id] = set(os.listdir(folder_path))
        else:
            phase_to_filenames[phase_id] = set()

    with open(annotations_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row.get("split", "train").strip().lower() != split_lower:
                continue

            try:
                phase_id = int(row["phase"])
            except (KeyError, ValueError):
                continue
            if not (0 <= phase_id < num_classes):
                continue

            image_name = row.get("image_name", "").strip()
            if not image_name:
                continue

            if image_name in phase_to_filenames[phase_id]:
                image_counts[phase_id] += 1

    fig, ax1 = plt.subplots(figsize=(15, 10))
    phases = list(range(num_classes))
    ax1.bar(phases, image_counts, edgecolor="black")
    ax1.set_title(f"Number of Images per Phase ({split.capitalize()} Split)")
    ax1.set_xlabel("Phase ID")
    ax1.set_ylabel("Number of Images")
    ax1.set_xticks(phases)
    ax1.set_xticklabels([str(p) for p in phases], rotation=45)

    for phase_id, count in enumerate(image_counts):
        if count > 0:
            ax1.text(phase_id, count, str(count), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"phase_images_distribution_{split_lower}.png")
    plt.savefig(save_path)
    plt.close(fig)

    total_images = sum(image_counts)
    phases_present = sum(1 for c in image_counts if c > 0)
    print(
        f"\nSummary for '{split}' split:\n"
        f"  • Total number of images: {total_images}\n"
        f"  • Number of phases present: {phases_present}\n"
    )

    return image_counts


if __name__ == "__main__":
    annotations_file = "dataset/annotations/image_annotations.csv"
    image_dir = "dataset/images"
    out_dir = "visualize/graphs"

    plot_phase_distribution(
        annotations_file, image_dir, out_dir, split="train"
    )
    plot_phase_distribution(
        annotations_file, image_dir, out_dir, split="val"
    )
