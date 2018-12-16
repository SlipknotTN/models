"""Convert the FASSEG dataset to TFRecord for object_detection API.
Example usage:
    python object_detection/dataset_tools/create_fasseg_tf_record.py \
        --data_dir=/home/user/FASSEG \
        --output_dir=/home/user/FASSEG/output
"""

import hashlib
import io
import logging
import os
import random
import json

import glob
import contextlib2
import PIL.Image
import tensorflow as tf
from tqdm import tqdm

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to FASSEG dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'data/pet_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_string('mask_type', 'png', 'How to represent instance '
                    'segmentation masks. Options are "png" or "numerical".')
flags.DEFINE_integer('num_shards', 1, 'Number of TFRecord shards')

FLAGS = flags.FLAGS


def retrieve_files(data_dir, super_directory):
    images = []
    _, dirs, _ = next(os.walk(os.path.join(data_dir, super_directory)))
    for dir in dirs:
        images.extend(glob.glob(os.path.join(data_dir, super_directory, dir) + "/*.*"))
    return images


def get_class_name_from_mask_path(mask_path):
  """
  Gets the class name from a mask path.

  Returns:
    A string of the class name.
  """
  filename = os.path.basename(mask_path)
  class_name = filename[filename.find("_") + 1 : filename.rfind("_")]
  return class_name


def dict_to_tf_example(data,
                       mask_paths,
                       bbox_paths,
                       label_map_dict,
                       mask_type='png'):
  """
  Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding extracted metadata from FASSEG
    mask_paths: PNG file paths with masks.
    bbox_paths: JSON file paths with bounding boxes.
    label_map_dict: A map from string label names to integers ids.
    mask_type: 'numerical' or 'png'. 'png' is recommended because it leads to smaller file sizes.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  img_path = data['filename']
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)

  width = image.size[0]
  height = image.size[1]

  if image.format != 'JPEG':
    encoded_jpg = io.BytesIO()
    image.save(encoded_jpg, format='JPEG')
    # Convert to bytes
    encoded_jpg.seek(0)
    encoded_jpg = encoded_jpg.read()

  key = hashlib.sha256(encoded_jpg).hexdigest()

  xmins = []
  ymins = []
  xmaxs = []
  ymaxs = []
  classes = []
  classes_text = []
  encoded_mask_png_list = []

  # Check gt existence (at least one png and json couple) and assert masks length == bbox length
  if len(mask_paths) != len(bbox_paths) or len(mask_paths) == 0:
    raise ValueError("Error in ground truth")

  for index, mask_path in enumerate(mask_paths):

    # Retrieve class name from mask_path
    class_name = get_class_name_from_mask_path(mask_path)
    classes_text.append(class_name.encode('utf8'))
    classes.append(label_map_dict[class_name])

    # Retrieve bounding boxes
    with open(bbox_paths[index], "r") as json_file:
      bbox = json.load(json_file)

    xmins.append(bbox["x_min"] / float(width))
    ymins.append(bbox["y_min"] / float(height))
    xmaxs.append(bbox["x_max"] / float(width))
    ymaxs.append(bbox["y_max"] / float(height))

    # Masks are already PNG encoded
    mask = PIL.Image.open(mask_path)
    if mask.format != 'PNG':
        raise ValueError('Mask format not PNG')
    # Convert to bytes
    encoded_png = io.BytesIO()
    mask.save(encoded_png, format='PNG')
    encoded_png.seek(0)
    encoded_png = encoded_png.read()
    encoded_mask_png_list.append(encoded_png)

  feature_dict = {
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),

  }

  if mask_type == 'numerical':
    raise Exception("mask_type numerical not supported with FASSEG dataset")
  elif mask_type == 'png':
    feature_dict['image/object/mask'] = (dataset_util.bytes_list_feature(encoded_mask_png_list))

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example


def create_tf_record(output_filename,
                     num_shards,
                     label_map_dict,
                     image_paths,
                     gts_paths,
                     mask_type='png'):
  """
  Creates a TFRecord file from examples.
  Args:
    output_filename: Path to where output file is saved.
    num_shards: Number of shards for output file.
    label_map_dict: The label map dictionary.
    image_paths: image files,
    gts_paths: ground truth files,
    mask_type: 'numerical' or 'png'. 'png' is recommended because it leads to smaller file sizes.
  """
  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_filename, num_shards)
    for idx, image_path in tqdm(enumerate(image_paths), desc="image"):
      if idx % 100 == 0:
        logging.info('On image %d of %d', idx, len(image_paths))

      # Much simpler data than PASCAL format retrieved from XML file
      data = dict()
      data["filename"] = image_path

      base_name = os.path.basename(image_path)[:-4]
      full_dir = os.path.dirname(image_path)
      sub_dir = full_dir[full_dir.rfind("/") + 1:]

      prefix = os.path.join(sub_dir, base_name)

      mask_paths = list()
      bbox_paths = list()

      for candidate_gt_file in gts_paths:
        if candidate_gt_file.find(prefix + "_") > 0:
          if candidate_gt_file[-3:] == "png":
            mask_paths.append(candidate_gt_file)
          elif candidate_gt_file[-4:] == "json":
            bbox_paths.append(candidate_gt_file)
          else:
            raise ValueError("Unknown ground truth file " + candidate_gt_file)

      mask_paths = sorted(mask_paths)
      bbox_paths = sorted(bbox_paths)

      try:
        tf_example = dict_to_tf_example(
            data,
            mask_paths,
            bbox_paths,
            label_map_dict,
            mask_type=mask_type)
        if tf_example:
          shard_idx = idx % num_shards
          output_tfrecords[shard_idx].write(tf_example.SerializeToString())
      except ValueError:
        logging.warning('Invalid example: %s, ignoring.', image_path)


def main(_):
  data_dir = FLAGS.data_dir
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  logging.info('Reading from FASSEG dataset.')

  # Retrieve list of images
  train_image_paths = retrieve_files(data_dir, "Train_RGB")
  val_image_paths = retrieve_files(data_dir, "Test_RGB")

  train_gts = retrieve_files(data_dir, "Train_labels_masks_instances")
  val_gts = retrieve_files(data_dir, "Test_labels_masks_instances")

  # We use the test subset as validation
  random.seed(42)
  random.shuffle(train_image_paths)
  random.shuffle(val_image_paths)
  logging.info('%d training and %d validation examples.', len(train_image_paths), len(val_image_paths))

  train_output_path = os.path.join(FLAGS.output_dir, 'fasseg_train.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'fasseg_val.record')

  create_tf_record(
      train_output_path,
      FLAGS.num_shards,
      label_map_dict,
      train_image_paths,
      train_gts,
      mask_type=FLAGS.mask_type)

  create_tf_record(
      val_output_path,
      FLAGS.num_shards,
      label_map_dict,
      val_image_paths,
      val_gts,
      mask_type=FLAGS.mask_type)


if __name__ == '__main__':
  tf.app.run()
