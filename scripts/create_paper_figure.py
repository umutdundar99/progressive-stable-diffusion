#!/usr/bin/env python3
"""
Create publication-ready figure grids for paper.
Combines MES progression images with proper labels.
"""

import argparse
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def find_closest_image(folder: Path, target_mes: float) -> tuple[Path | None, float]:
    """Find the image with MES value closest to target."""
    best_match = None
    best_diff = float("inf")
    best_mes = None

    for img_path in folder.glob("mes_*.png"):
        if img_path.name in ["progression_grid.png", "structure_reference.png"]:
            continue

        # Parse MES value from filename: mes_0.00_00.png
        parts = img_path.stem.split("_")
        if len(parts) >= 2:
            try:
                mes_val = float(parts[1])
                diff = abs(mes_val - target_mes)
                if diff < best_diff:
                    best_diff = diff
                    best_match = img_path
                    best_mes = mes_val
            except ValueError:
                continue

    return best_match, best_mes


def get_font(size: int):
    """Get a font, falling back to default if custom fonts unavailable."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue

    # Fallback to default
    return ImageFont.load_default()


def create_horizontal_grid(
    input_folder: str,
    output_path: str,
    target_mes_values: list[float] = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    image_size: int = 256,
    padding: int = 10,
    label_height: int = 40,
    title_height: int = 50,
    title: str = None,
    font_size: int = 24,
    title_font_size: int = 28,
    show_actual_values: bool = False,
    background_color: tuple = (255, 255, 255),
):
    """
    Create a horizontal grid of MES progression images.

    Args:
        input_folder: Path to folder containing mes_*.png images
        output_path: Path for output figure
        target_mes_values: List of target MES values to include
        image_size: Size of each image (assuming square)
        padding: Padding between images
        label_height: Height for labels below images
        title_height: Height for title above images
        title: Optional title for the figure
        font_size: Font size for labels
        title_font_size: Font size for title
        show_actual_values: Whether to show actual MES values in parentheses
        background_color: Background color (RGB tuple)
    """
    folder = Path(input_folder)

    # Find images for each target MES value
    selected_images = []
    for target in target_mes_values:
        img_path, actual_mes = find_closest_image(folder, target)
        if img_path is not None:
            selected_images.append(
                {"path": img_path, "target": target, "actual": actual_mes}
            )
            print(f"  MES {target:.1f} -> {img_path.name} (actual: {actual_mes:.2f})")
        else:
            print(f"  MES {target:.1f} -> NOT FOUND")

    if not selected_images:
        raise ValueError(f"No images found in {folder}")

    n_images = len(selected_images)

    # Calculate canvas size
    total_width = n_images * image_size + (n_images + 1) * padding
    total_height = image_size + label_height + 2 * padding
    if title:
        total_height += title_height

    # Create canvas
    canvas = Image.new("RGB", (total_width, total_height), background_color)
    draw = ImageDraw.Draw(canvas)

    # Load fonts
    label_font = get_font(font_size)
    title_font = get_font(title_font_size)

    # Draw title if provided
    y_offset = padding
    if title:
        # Center title
        bbox = draw.textbbox((0, 0), title, font=title_font)
        text_width = bbox[2] - bbox[0]
        title_x = (total_width - text_width) // 2
        draw.text((title_x, y_offset), title, fill=(0, 0, 0), font=title_font)
        y_offset += title_height

    # Place images and labels
    for i, img_info in enumerate(selected_images):
        x = padding + i * (image_size + padding)

        # Load and resize image
        img = Image.open(img_info["path"]).convert("RGB")
        if img.size != (image_size, image_size):
            img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)

        # Paste image
        canvas.paste(img, (x, y_offset))

        # Create label
        if show_actual_values:
            label = f"MES {img_info['target']:.1f}\n({img_info['actual']:.2f})"
        else:
            label = f"MES {img_info['target']:.1f}"

        # Center label below image
        bbox = draw.textbbox((0, 0), label, font=label_font)
        text_width = bbox[2] - bbox[0]
        label_x = x + (image_size - text_width) // 2
        label_y = y_offset + image_size + 5

        draw.text((label_x, label_y), label, fill=(0, 0, 0), font=label_font)

    # Save
    canvas.save(output_path, dpi=(300, 300))
    print(f"\nSaved: {output_path}")
    print(f"Size: {total_width} x {total_height} pixels")

    return canvas


def create_comparison_grid(
    input_folders: list[str],
    model_names: list[str],
    output_path: str,
    target_mes_values: list[float] = [0.0, 1.0, 2.0, 3.0],
    image_size: int = 200,
    padding: int = 8,
    label_height: int = 35,
    row_label_width: int = 120,
    font_size: int = 20,
    background_color: tuple = (255, 255, 255),
):
    """
    Create a comparison grid with multiple models (rows) and MES values (columns).

    Args:
        input_folders: List of folders containing mes_*.png images for each model
        model_names: List of model names for row labels
        output_path: Path for output figure
        target_mes_values: List of target MES values for columns
        image_size: Size of each image
        padding: Padding between images
        label_height: Height for column labels
        row_label_width: Width for row labels
        font_size: Font size for labels
        background_color: Background color
    """
    n_rows = len(input_folders)
    n_cols = len(target_mes_values)

    # Calculate canvas size
    total_width = row_label_width + n_cols * image_size + (n_cols + 1) * padding
    total_height = label_height + n_rows * image_size + (n_rows + 1) * padding

    # Create canvas
    canvas = Image.new("RGB", (total_width, total_height), background_color)
    draw = ImageDraw.Draw(canvas)

    # Load fonts
    font = get_font(font_size)

    # Draw column headers (MES values)
    for j, mes_val in enumerate(target_mes_values):
        label = f"MES {mes_val:.1f}"
        x = row_label_width + padding + j * (image_size + padding)

        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        label_x = x + (image_size - text_width) // 2

        draw.text((label_x, padding), label, fill=(0, 0, 0), font=font)

    # Draw each row
    for i, (folder, model_name) in enumerate(zip(input_folders, model_names)):
        folder = Path(folder)
        y = label_height + padding + i * (image_size + padding)

        # Draw row label (model name)
        bbox = draw.textbbox((0, 0), model_name, font=font)
        text_height = bbox[3] - bbox[1]
        label_y = y + (image_size - text_height) // 2
        draw.text((padding, label_y), model_name, fill=(0, 0, 0), font=font)

        # Draw images for this row
        for j, target_mes in enumerate(target_mes_values):
            x = row_label_width + padding + j * (image_size + padding)

            img_path, actual_mes = find_closest_image(folder, target_mes)
            if img_path is not None:
                img = Image.open(img_path).convert("RGB")
                if img.size != (image_size, image_size):
                    img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                canvas.paste(img, (x, y))
            else:
                # Draw placeholder
                draw.rectangle(
                    [x, y, x + image_size, y + image_size], outline=(200, 200, 200)
                )
                draw.text(
                    (x + 10, y + image_size // 2),
                    "N/A",
                    fill=(150, 150, 150),
                    font=font,
                )

    # Save
    canvas.save(output_path, dpi=(300, 300))
    print(f"\nSaved: {output_path}")
    print(f"Size: {total_width} x {total_height} pixels")

    return canvas


def create_single_row_with_reference(
    input_folder: str,
    output_path: str,
    target_mes_values: list[float] = [0.0, 1.0, 2.0, 3.0],
    image_size: int = 256,
    padding: int = 15,
    label_height: int = 45,
    font_size: int = 26,
    include_reference: bool = True,
    reference_label: str = "Reference",
    background_color: tuple = (255, 255, 255),
    add_arrow: bool = True,
):
    """
    Create a single row figure with optional reference image and MES progression.
    """
    folder = Path(input_folder)

    # Find reference image
    ref_path = folder / "structure_reference.png"
    has_reference = include_reference and ref_path.exists()

    # Find MES images
    selected_images = []
    for target in target_mes_values:
        img_path, actual_mes = find_closest_image(folder, target)
        if img_path is not None:
            selected_images.append(
                {"path": img_path, "target": target, "actual": actual_mes}
            )

    n_images = len(selected_images) + (1 if has_reference else 0)

    # Add space for arrow
    arrow_width = 40 if add_arrow and has_reference else 0

    # Calculate canvas size
    total_width = n_images * image_size + (n_images + 1) * padding + arrow_width
    total_height = image_size + label_height + 2 * padding

    # Create canvas
    canvas = Image.new("RGB", (total_width, total_height), background_color)
    draw = ImageDraw.Draw(canvas)

    font = get_font(font_size)
    x_pos = padding

    # Draw reference image
    if has_reference:
        ref_img = Image.open(ref_path).convert("RGB")
        if ref_img.size != (image_size, image_size):
            ref_img = ref_img.resize((image_size, image_size), Image.Resampling.LANCZOS)
        canvas.paste(ref_img, (x_pos, padding))

        # Label
        bbox = draw.textbbox((0, 0), reference_label, font=font)
        text_width = bbox[2] - bbox[0]
        label_x = x_pos + (image_size - text_width) // 2
        draw.text(
            (label_x, padding + image_size + 5),
            reference_label,
            fill=(0, 0, 0),
            font=font,
        )

        x_pos += image_size + padding

        # Draw arrow
        if add_arrow:
            arrow_y = padding + image_size // 2
            arrow_start = x_pos
            arrow_end = x_pos + arrow_width - 10

            # Arrow line
            draw.line(
                [(arrow_start, arrow_y), (arrow_end, arrow_y)],
                fill=(100, 100, 100),
                width=3,
            )
            # Arrow head
            draw.polygon(
                [
                    (arrow_end, arrow_y),
                    (arrow_end - 10, arrow_y - 8),
                    (arrow_end - 10, arrow_y + 8),
                ],
                fill=(100, 100, 100),
            )

            x_pos += arrow_width

    # Draw MES images
    for img_info in selected_images:
        img = Image.open(img_info["path"]).convert("RGB")
        if img.size != (image_size, image_size):
            img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
        canvas.paste(img, (x_pos, padding))

        # Label
        label = f"MES {img_info['target']:.1f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        label_x = x_pos + (image_size - text_width) // 2
        draw.text((label_x, padding + image_size + 5), label, fill=(0, 0, 0), font=font)

        x_pos += image_size + padding

    # Save
    canvas.save(output_path, dpi=(300, 300))
    print(f"\nSaved: {output_path}")
    print(f"Size: {total_width} x {total_height} pixels")

    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="Create paper figures from MES progression images"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input folder with mes_*.png images",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output path for figure"
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["horizontal", "reference", "comparison"],
        default="reference",
        help="Figure mode",
    )
    parser.add_argument(
        "--mes-values",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        help="Target MES values to include",
    )
    parser.add_argument("--image-size", type=int, default=256, help="Image size")
    parser.add_argument(
        "--font-size", type=int, default=26, help="Font size for labels"
    )
    parser.add_argument("--title", type=str, default=None, help="Figure title")
    parser.add_argument(
        "--no-reference", action="store_true", help="Exclude reference image"
    )
    parser.add_argument(
        "--no-arrow",
        action="store_true",
        help="Exclude arrow between reference and progression",
    )

    args = parser.parse_args()

    # Set default output path
    if args.output is None:
        input_name = Path(args.input).name
        args.output = f"paper_figure_{input_name}.png"

    print(f"Creating {args.mode} figure...")
    print(f"Input: {args.input}")
    print(f"Target MES values: {args.mes_values}")

    if args.mode == "horizontal":
        create_horizontal_grid(
            input_folder=args.input,
            output_path=args.output,
            target_mes_values=args.mes_values,
            image_size=args.image_size,
            font_size=args.font_size,
            title=args.title,
        )
    elif args.mode == "reference":
        create_single_row_with_reference(
            input_folder=args.input,
            output_path=args.output,
            target_mes_values=args.mes_values,
            image_size=args.image_size,
            font_size=args.font_size,
            include_reference=not args.no_reference,
            add_arrow=not args.no_arrow,
        )


if __name__ == "__main__":
    main()
