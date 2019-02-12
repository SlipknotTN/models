import argparse
import os

import tensorflow as tf
from tqdm import tqdm

from distutils.version import StrictVersion
from matplotlib import pyplot as plt
from PIL import Image

# The script must be launched from repository root
from research.object_detection.utils import label_map_util
from research.object_detection.utils import visualization_utils as vis_util

from libs.utils.utils import load_image_into_numpy_array
from libs.inference.run_inference import run_inference_for_single_image_with_graph


if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


def do_parsing():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Images directory inference script")
    parser.add_argument("--dataset_dir", required=True, type=str, help="Dataset test directory")
    parser.add_argument("--model_path", required=True, type=str, help="Filepath with trained model")
    parser.add_argument("--labels_path", required=False, type=str,
                        default="research/object_detection/data/mscoco_label_map.pbtxt",
                        help="Path to pbtxt labels file description which depends on the dataset")
    parser.add_argument("--output_dir", required=False, type=str, help="Export directory for predictions")
    parser.add_argument("--show", action="store_true", help="Show results on GUI")

    args = parser.parse_args()
    return args


def main():
    """
    TF Object Detection API Installation:
    https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
    """
    args = do_parsing()
    print(args)

    assert os.path.exists(args.dataset_dir), "Dataset directory does not exist"

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=False)

    # Load labels
    category_index = label_map_util.create_category_index_from_labelmap(args.labels_path, use_display_name=True)

    # Load model
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(args.model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Retrieve list of test images
    extensions = ["jpg", "png", "bmp"]
    imagefiles = []
    for root, dirs, files in os.walk(args.dataset_dir):
        for extension in extensions:
            imagefiles.extend(os.path.join(root, file) for file in files if file.endswith("." + extension))

    # Run predictions

    # Size, in inches, of the output images for matplotlib
    IMAGE_SIZE = (12, 8)

    for image_path in tqdm(imagefiles, desc="image"):
        print(image_path)
        image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # Actual detection
        output_dict = run_inference_for_single_image_with_graph(image_np, detection_graph)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            min_score_thresh=0.7,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=2)
        plt.figure(figsize=IMAGE_SIZE)
        # plt.imshow(image_np)
        if args.output_dir:
            plt.imsave(os.path.join(args.output_dir, os.path.basename(image_path)), image_np)


if __name__ == "__main__":
    main()
