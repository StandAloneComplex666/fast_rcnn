import tensorflow as tf
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from PIL import Image
from pathlib import Path
import os
import io
import random
import sys

from xml_to_dict import Xml_to_dict


# File Paths
xml_dir = "Annotations"
images_dir = "Images"


# From the original TFOD
flags = tf.app.flags
flags.DEFINE_string('output_dir', '', 'Output directory')
flags.DEFINE_string('label_map_path', '', 'Path to label map proto')
FLAGS = flags.FLAGS

# Additional added
x2d = Xml_to_dict()  # For bounding box values
random.seed(42)     # This creates repeatable datasets


def create_tf_example(classDict, imgFilename, imgLabels, imgBBs):
    with open(imgFilename, 'rb') as fid:
        # with tf.gfile.GFile(imgFilename, 'rb') as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    image_format = image.format
    # if image_format is not 'jpg' or not 'png':
    #     raise ValueError('{}: Image format not JPEG or PNG'.format(imgFilename))
    #     return None

    filename = imgFilename  # Filename of the image. Empty if image is not from file
    encoded_image_data = encoded_jpg  # Encoded image bytes

    xmins = [float(bb[0][0]) / width for bb in imgBBs]  # List of normalized left x coordinates in bounding box (1 per box)
    xmins = list(map(lambda x: min(x, 1.), xmins))  # clip value to 1
    xmins = list(map(lambda x: max(x, 0.), xmins))  # clip value to 0

    xmaxs = [float(bb[1][0]) / width for bb in imgBBs]  # List of normalized right x coordinates in bounding box (1 per box)
    xmaxs = list(map(lambda x: min(x, 1.), xmaxs))  # clip value to 1
    xmaxs = list(map(lambda x: max(x, 0.), xmaxs))  # clip value to 0

    for i in range(len(xmins)):
        if xmins[i] > xmaxs[i]:
            #print(xmins[i], xmaxs[i])
            xmins[i], xmaxs[i] = xmaxs[i], xmins[i]
    #print ('xmins:',xmins,'  xmaxs:',xmaxs)
    ymins = [float(bb[0][1]) / height for bb in imgBBs]  # List of normalized top y coordinates in bounding box (1 per box)
    ymins = list(map(lambda x: min(x, 1.), ymins))  # clip value to 1
    ymins = list(map(lambda x: max(x, 0.), ymins))  # clip value to 0

    ymaxs = [float(bb[1][1]) / height for bb in imgBBs]  # List of normalized bottom y coordinates in bounding box (1 per box)
    ymaxs = list(map(lambda x: min(x, 1.), ymaxs))  # clip value to 1
    ymaxs = list(map(lambda x: max(x, 0.), ymaxs))  # clip value to 0
    for i in range(len(ymins)):
        if ymins[i] > ymaxs[i]:
            #print(ymins[i], ymaxs[i])
            ymins[i], ymaxs[i] = ymaxs[i], ymins[i]
    assert max(xmins) < 1.01 and max(xmaxs) < 1.01 and max(ymins) < 1.01 and max(ymaxs) < 1.01, "One or more bb normalized coordinates > 1.01 in {}".format(filename)
    assert min(xmins) >= 0.0 and min(xmaxs) >= 0.0 and min(ymins) >= 0.0 and min(ymaxs) >= 0.0, "One or more bb normalized coordinates < 0.00 in {}".format(filename)

    classes_text = imgLabels  # List of string class name of bounding box (1 per box)
    # print(imgLabels)
    classes = [classDict[lbl] for lbl in imgLabels]  # List of integer class id of bounding box (1 per box)

    # Encoding to bytes
    filename_bytes = filename.encode()
    image_format_bytes = image_format.encode()
    class_text_bytes = []

    for class_obj in classes_text:
        class_text_bytes.append(class_obj.encode())
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename_bytes),
            'image/source_id': dataset_util.bytes_feature(filename_bytes),
            'image/encoded': dataset_util.bytes_feature(encoded_image_data),
            'image/format': dataset_util.bytes_feature(image_format_bytes),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(class_text_bytes),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))

    return tf_example


def belongsToTrainingSet():
    if random.uniform(0, 1) <= 0.8:
        return True
    return False


def main(_):

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    label_map_dict_rev = dict((v, k) for k, v in label_map_dict.items())

    print(label_map_dict)

    train_output_path = os.path.join(FLAGS.output_dir, 'FL_train_rand.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'FL_val_rand.record')

    train_lst_file_path = os.path.join(FLAGS.output_dir, 'train_file_list.txt')
    val_lst_file_path = os.path.join(FLAGS.output_dir, 'val_file_list.txt')

    writer_train = tf.python_io.TFRecordWriter(train_output_path)
    writer_val = tf.python_io.TFRecordWriter(val_output_path)

    train_file = open(train_lst_file_path, 'w')
    val_file = open(val_lst_file_path, 'w')

    count = 1
    count_train = 0
    count_val = 0

    missingImgFiles = []
    nonJPGFiles = []
    allImagesList = []

    # Compare if images in directory match with the xml files
    for subdir in os.listdir(images_dir):
        subdir_path = os.path.join(images_dir, subdir)
        for filename in os.listdir(subdir_path):
            imgFilename = os.path.join(images_dir, subdir, filename)
            xml_file_name = filename.split('.')[0] + '.xml'
            xml_path = os.path.join(xml_dir, subdir, xml_file_name)
            if os.path.isfile(xml_path):
                allImagesList.append(imgFilename)
            else:
                missingImgFiles.append(imgFilename)

    random.shuffle(allImagesList)

    for imgFilename in allImagesList:
        parts = os.path.split(imgFilename)
        xml_file_name = parts[-1].split('.')[0] + '.xml'    # ABC/XYZ/pqr.jpg ->  pqr.jpg ->pqr.xml
        subdir = os.path.split(parts[0])[-1]                # ABC/XYZ -> XYZ
        xml_path = os.path.join(xml_dir, subdir, xml_file_name)
        try:
            bbLabels, bbCords = x2d.label_coordinates(xml_path)
        except:
            continue
        # print(bbLabels, bbCords)
        # print(xml_path)
        # print(imgFilename)
        if len(bbLabels) != 0:
            try:
                tf_example = create_tf_example(label_map_dict, imgFilename, bbLabels, bbCords)
            except:
                print(sys.exc_info()[0])
                continue

            if tf_example is not None:
                if belongsToTrainingSet() is True:
                    count_train += 1
                    writer_train.write(tf_example.SerializeToString())
                    train_file.write(imgFilename + '\n')
                else:
                    count_val += 1
                    writer_val.write(tf_example.SerializeToString())
                    val_file.write(imgFilename + '\n')
            else:
                nonJPGFiles.append(imgFilename)

        if count % 100 == 0:
            print("Processed {} files...".format(count))
            # break
        # print(count)
        count += 1

    print("Done!")
    print ("Total number of annotation files processed: {}".format(count))
    print ("Size of training set: {}".format(count_train))
    print ("Size of validation set: {}".format(count_val))
    print ("Size of training+ val sets: {}".format(count_train + count_val))
    print ("{} Missing image files: ".format(len(missingImgFiles)))
    # print (missingImgFiles)
    print ("{} non-JPG image files: ".format(len(nonJPGFiles)))
    print (nonJPGFiles)

    writer_train.close()
    writer_val.close()
    train_file.close()
    val_file.close()


if __name__ == '__main__':
    tf.app.run()
