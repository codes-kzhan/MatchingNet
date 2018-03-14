"""Utils to prepare data"""
import numpy as np
import tensorflow as tf
from scipy import misc
import mobilenet_v1
import os
from tensorflow.python import pywrap_tensorflow
from PIL import Image


class DataUtils:

    def __init__(self, data_path):

        """
        Get small batch from saved feature maps
        :param data_path: 
        """
        data = np.load(data_path)
        self.feature_maps = data["feature_maps"]
        self.labels = data["labels"]

    def get_batch(self, possible_classes=5, shot=5, batch_size=32):

        support_features = []
        support_labels = []
        target_feature = []
        target_label = []

        for i in range(batch_size):
            class_index = np.random.choice(len(self.feature_maps), possible_classes, False)
            target_class = np.random.randint(possible_classes)

            support_feature = []
            support_label = []
            for j in range(possible_classes):
                feature_map = self.feature_maps[class_index[j]]

                if j == target_class:
                    file_index = np.random.choice(len(feature_map), shot + 1, False)
                    file_index, target_file_index = np.split(file_index, [shot])

                    target_feature.append(feature_map[target_file_index[0]])
                    target_label.append(target_class)
                else:
                    file_index = np.random.choice(len(feature_map), shot, False)

                support_feature.extend(feature_map[file_index])
                support_label.extend([j]*shot)

            support_features.append(support_feature)
            support_labels.append(support_label)

        return support_features, support_labels, target_feature, target_label


def search_feature(data_path):

    """
    Searching data in all subdirs
    :param data_path: Dir containing images
    :return: feature paths and labels
    """

    print("Searching ", data_path)

    roots = []
    dirs_all = []
    files_all = []
    for root, dirs, files in os.walk(data_path):
        roots.append(root)
        dirs_all.append(dirs)
        if len(files) > 1:
            files_all.append([os.path.join(root, file) for file in files])

    labels = dirs_all[0]
    class_num = len(labels)

    feature_paths = files_all
    file_num = sum([len(file_paths) for file_paths in feature_paths])

    print("Find " + str(class_num) + " classes")
    print("Find " + str(file_num) + " files")

    return feature_paths, labels


def get_feature_map(data_path, save_path, model_dir,
                    model_name="MOBILENET", pic_size=224, channel=3):

    """
    
    :param data_path: path of data to be encoded
    :param save_path: 
    :param model_name: base model or model to encode data
    :param model_dir: checkpoint dir
    :param pic_size: picture size
    :param channel: channel of picture
    :param old_data_path: 
    :return: 
    """
    features = tf.placeholder(tf.float32, shape=[None, pic_size, pic_size, channel])

    if model_name == "MOBILENET":

        # net, endpoints = mobilenet_v1.mobilenet_v1(features, scope='FeatureExtractor/MobilenetV1')
        net, endpoints = mobilenet_v1.mobilenet_v1(features)

        logits = endpoints['Logits']
        # logits = tf.contrib.layers.flatten(logits)
        # logits = endpoints['AvgPool_1a']

    elif model_name == "MNIST_MODEL":

        conv1 = tf.layers.conv2d(
            inputs=features,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

        logits = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    else:
        print("No such model type")
        return -1

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if os.path.isdir(model_dir):
        module_file = tf.train.latest_checkpoint(model_dir)
    else:
        module_file = model_dir

    print("Load checkpoint from ", module_file)
    reader = pywrap_tensorflow.NewCheckpointReader(module_file)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name: ", key)
    restorer = tf.train.Saver()
    restorer.restore(sess, module_file)
    print("Done")

    feature_maps = []

    if data_path[-4:] == ".npz":

        data = np.load(data_path)
        img_data = data["features"]
        img_data = img_data.reshape((-1, 20, pic_size, pic_size, channel))
        labels = data["labels"]

        for img in img_data:
            feed_dict = {features: img}
            feature_map = sess.run(logits, feed_dict=feed_dict)
            feature_maps.append(feature_map)

    else:

        feature_paths, labels = search_feature(data_path)

        for i, paths in enumerate(feature_paths):
            print("Class ", labels[i])
            imgs = []
            for path in paths:
                print("Reading file ", path)
                img = misc.imread(path)
                img_resized = misc.imresize(img, [pic_size, pic_size])
                if len(img_resized.shape) == 2:
                    img_resized = np.stack([img_resized] * 3, 2)
                imgs.append(img_resized)

            print("Processing...")
            feed_dict = {features: imgs}
            feature_map = sess.run(logits, feed_dict=feed_dict)
            feature_maps.append(feature_map)

    print("Saving  ", save_path)
    np.savez(save_path, feature_maps=np.squeeze(feature_maps), labels=labels)
    print("DONE")


# get_feature_map("dataset/tiny-imagenet-200/validation",
#                 "dataset/feature_maps/mobilenet/tiny_val.npz",
#                 "tmp/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt")
def generate_tf_record(data_path, save_path, pic_size=224, channel=3):

    feature_paths, labels = search_feature(data_path)

    writer = tf.python_io.TFRecordWriter(save_path)

    for index, paths in feature_paths:
        for path in paths:
            img = Image.open(path)
            img = img.resize((pic_size, pic_size))
            img = img.convert('RGB')
            img_raw = img.tobytes()

            example = tf.train.Example(
                feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    "feature": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }
            )

            writer.write(example.SerializeToString())

    writer.close()


def read_tf_record(file_path):

    file_queue = tf.train.string_input_producer([file_path])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           "label": tf.FixedLenFeature([], tf.int64),
                                           "feature": tf.FixedLenFeature([], tf.string)
                                       })
    img = tf.decode_raw(features["feature"], tf.uint8)
    img = tf.reshape(img, [128, 128, 3])