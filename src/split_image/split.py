import argparse
import os
import re
from collections import Counter
from typing import List, Union

import torch
from PIL import Image

def split_image(image_path: str, rows: int, cols: int, should_square: bool, should_cleanup: bool, should_quiet: bool = False, output_dir: str = None) -> List[Image.Image]:
    if not os.path.isfile(image_path):
        raise ValueError(f"Invalid image path: {image_path}")    
    try:
        im = Image.open(image_path)
    except IOError:
        raise ValueError(f"Unable to open image file: {image_path}")    
    im_width, im_height = im.size
    row_width = int(im_width / cols)
    row_height = int(im_height / rows)
    name, ext = os.path.splitext(image_path)
    name = os.path.basename(name)
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "ComfyUI", "output", "Comfly_split_image")
    else:
        os.makedirs(output_dir, exist_ok=True)

    split_images: List[Image.Image] = []
    n = 0
    for i in range(rows):
        for j in range(cols):
            box = (j * row_width, i * row_height, j * row_width + row_width, i * row_height + row_height)
            outp = im.crop(box)
            split_images.append(outp)
            output_path = os.path.join(output_dir, f"{name}_{n}{ext}")  
            if not should_quiet:
                print(f"Exporting image tile: {output_path}")
            outp.save(output_path)
            n += 1

    if should_cleanup:
        if not should_quiet:
            print("Cleaning up: " + image_path)
        os.remove(image_path)
    
    print(f"Split images: {split_images}")  # 打印分割后的图片列表
    print(f"Number of split images: {len(split_images)}")  # 打印分割后的图片数量
    return split_images  # 返回分割后的图片列表

def reverse_split(paths_to_merge: List[str], rows: int, cols: int, image_path: str, should_cleanup: bool, should_quiet: bool = False):
    if not paths_to_merge:
        print("No images to merge!")
        return

    for index, path in enumerate(paths_to_merge):
        path_number = int(path.split("_")[-1].split(".")[0])
        if path_number != index:
            print(f"Warning: Image {path} has a number that does not match its index!")
            print("Please rename it first to match the rest of the images.")
            return

    images_to_merge = [Image.open(p) for p in paths_to_merge]
    image1 = images_to_merge[0]
    new_width = image1.size[0] * cols
    new_height = image1.size[1] * rows

    if not should_quiet:
        print("Merging image tiles with the following layout:", end=" ")
        for i in range(rows):
            print("\n")
            for j in range(cols):
                print(paths_to_merge[i * cols + j], end=" ")
        print("\n")

    new_image = Image.new(image1.mode, (new_width, new_height))
    for i in range(rows):
        for j in range(cols):
            image = images_to_merge[i * cols + j]
            new_image.paste(image, (j * image.size[0], i * image.size[1]))

    if not should_quiet:
        print("Saving merged image: " + image_path)
    new_image.save(image_path)

    if should_cleanup:
        for p in paths_to_merge:
            if not should_quiet:
                print("Cleaning up: " + p)
            os.remove(p)

def determine_bg_color(im: Image.Image) -> tuple:
    im_width, im_height = im.size
    rgb_im = im.convert('RGBA')
    all_colors = []
    areas = [((0, 0), (im_width, im_height // 10)),
             ((0, 0), (im_width // 10, im_height)),
             ((im_width * 9 // 10, 0), (im_width, im_height)),
             ((0, im_height * 9 // 10), (im_width, im_height))]
    for area in areas:
        start = area[0]
        end = area[1]
        for x in range(int(start[0]), int(end[0])):
            for y in range(int(start[1]), int(end[1])):
                pix = rgb_im.getpixel((x, y))
                all_colors.append(pix)
    return Counter(all_colors).most_common(1)[0][0]

def pil2tensor(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    if isinstance(image, list):
        return torch.cat([pil2tensor(img) for img in image], dim=0)

    return torch.from_numpy(image.to_numpy().astype('float32') / 255.0).permute(2, 0, 1).unsqueeze(0)

def main():
    parser = argparse.ArgumentParser(description="Split an image into rows and columns.")
    parser.add_argument("image_path", nargs=1, help="The path to the image or directory with images to process.")
    parser.add_argument("rows", type=int, default=2, nargs='?', help="How many rows to split the image into (horizontal split).")
    parser.add_argument("cols", type=int, default=2, nargs='?', help="How many columns to split the image into (vertical split).")
    parser.add_argument("-s", "--square", action="store_true", help="If the image should be resized into a square before splitting.")
    parser.add_argument("-r", "--reverse", action="store_true", help="Reverse the splitting process, i.e. merge multiple tiles of an image into one.")
    parser.add_argument("--cleanup", action="store_true", help="After splitting or merging, delete the original image/images.")
    parser.add_argument("--load-large-images", action="store_true", help="Ignore the PIL decompression bomb protection and load all large files.")
    parser.add_argument("--output-dir", type=str, help="Set the output directory for image tiles (e.g. 'outp/images'). Defaults to current working directory.")
    parser.add_argument("--quiet", action="store_true", help="Run without printing any messages.")

    args = parser.parse_args()
    if args.load_large_images:
        Image.MAX_IMAGE_PIXELS = None
    image_path = args.image_path[0]
    if not os.path.exists(image_path):
        print("Error: Image path does not exist!")
        return

    if os.path.isdir(image_path):
        if args.reverse:
            print("Error: Cannot reverse split a directory of images!")
            return
        if not args.quiet:
            print("Splitting all images in directory: " + image_path)
        for file in os.listdir(image_path):
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                split_images = split_image(os.path.join(image_path, file), args.rows, args.cols, args.square, args.cleanup, args.quiet, args.output_dir)
                for i, img in enumerate(split_images):
                    output_path = os.path.join(output_dir, f"{name}_{i}{ext}")
                    if not args.quiet:
                        print(f"Saving split image: {output_path}")
                    img.save(output_path)
    else:
        if args.reverse:
            if not args.quiet:
                print("Reverse mode selected! Will try to merge multiple tiles of an image into one.\n")
            start_name, ext = os.path.splitext(image_path)
            expr = re.compile(r"^" + start_name + "_\d+" + ext + "$")
            paths_to_merge = sorted([f for f in os.listdir(os.getcwd()) if re.match(expr, f)], key=lambda x: int(x.split("_")[-1].split(".")[0]))
            reverse_split(paths_to_merge, args.rows, args.cols, image_path, args.cleanup, args.quiet)
        else:
            split_images = split_image(image_path, args.rows, args.cols, args.square, args.cleanup, args.quiet, args.output_dir)
            for i, img in enumerate(split_images):
                name, ext = os.path.splitext(image_path)
                name = os.path.basename(name)
                output_path = os.path.join(args.output_dir, f"{name}_{i}{ext}")
                if not args.quiet:
                    print(f"Saving split image: {output_path}")
                img.save(output_path)

    if not args.quiet:
        print("Done!")

if __name__ == "__main__":
   main()