import os
import argparse

import csv
import tensorflow as tf
from PIL import Image

from tqdm import tqdm

from libs.utils.utils import load_image_into_numpy_array
from libs.inference.run_inference import run_inference_for_single_image_with_session
from libs.drawing.drawing import draw_masks_on_single_image


def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--images_dir", required=True, type=str, help="Images directory")
    parser.add_argument("--model_path", required=False, type=str, help="H5 model file path")
    parser.add_argument("--output_dir", required=False, type=str, help="Output directory with predictions")
    parser.add_argument("--output_format", required=False, type=str, default="jpg", help="Mask images output extension")
    parser.add_argument("--classes_csv", required=True, type=str, help="CSV file with class-colors mapping")
    parser.add_argument("--threshold", required=False, type=float, default=0.7, help="Minimum threshold for classes")
    parser.add_argument("--debug", action="store_true", help="Export single predictions")
    args = parser.parse_args()
    return args


def main():

    args = do_parsing()
    print(args)

    # Read classes
    classes = []

    with open(args.classes_csv, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        # Skip header
        next(reader)
        for row in reader:
            colors = row[1].split(",")
            colors = tuple([int(color) for color in colors])
            # Convert to RGB
            colors = colors[::-1]
            assert len(colors) == 3, "Wrong number of colors for " + row[0]
            classes.append((row[0], colors))

    # Load model
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(args.model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Load a random image from the images folder
    exts = ["bmp", "jpg", "png"]

    os.makedirs(args.output_dir, exist_ok=True)

    images_paths = []

    for root, subdirs, files in os.walk(args.images_dir):
        for file in files:
            for ext in exts:
                if file.endswith("." + ext):
                    images_paths.append(os.path.join(root, file))
                    os.makedirs(os.path.join(args.output_dir, os.path.relpath(root, args.images_dir)), exist_ok=True)

    with detection_graph.as_default():

        with tf.Session() as sess:

            for image_path in tqdm(images_paths, desc="image"):
                print(image_path)
                image = Image.open(image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Actual detection
                output_dict = run_inference_for_single_image_with_session(image_np, sess)

                # Single image mask
                image_mask = draw_masks_on_single_image(output_dict, classes, threshold=args.threshold)

                dest_path = os.path.join(args.output_dir, os.path.relpath(image_path, args.images_dir))
                dest_full_pred_path = dest_path[:-3] + args.output_format
                output_image = Image.fromarray(image_mask)
                output_image.save(dest_full_pred_path)

    print("Success")


if __name__ == "__main__":
    main()