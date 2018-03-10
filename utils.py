import numpy as np
import tensorflow as tf
from scipy import misc
import mobilenet_v1
import os


class DataUtils:

    def __init__(self, data_path):

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


def get_feature_map(data_path, save_path, model_name="MOBILENET", model_dir="tmp/mobilenet_matchnet",
                    pic_size=224, channel=3, old_data_path = None):

    features = tf.placeholder(tf.float32, shape=[None, pic_size, pic_size, channel])

    if model_name == "MOBILENET":

        net, endpoints = mobilenet_v1.mobilenet_v1(features, prediction_fn=None,
                                          reuse=tf.AUTO_REUSE, scope='FeatureExtractor/MobilenetV1')

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
        return -1

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    module_file = tf.train.latest_checkpoint(model_dir)
    print("Load checkpoint from ", module_file)
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

        print("Searching pictures...")

        roots = []
        dirs_all = []
        files_all = []
        for root, dirs, files in os.walk(data_path):
            roots.append(root)
            dirs_all.append(dirs)
            files_all.append([os.path.join(root, file) for file in files])

        labels = dirs_all[0]
        class_num = len(labels)

        feature_path = files_all[1:(class_num + 1)]
        file_num = sum([len(file_paths) for file_paths in feature_path])

        print("Find " + str(class_num) + " classes")
        print("Find " + str(file_num) + " files")

        for paths in feature_path:
            imgs = []
            for path in paths:
                img = misc.imread(path)
                img_resized = misc.imresize(img, [pic_size, pic_size])
                img_resized = img_resized.reshape([pic_size, pic_size, channel])
                imgs.append(img_resized)

            feed_dict = {features: imgs}
            feature_map = sess.run(logits, feed_dict=feed_dict)
            feature_maps.append(feature_map)

    # if old_data_path is not None:
    #     old_data = np.load(old_data_path)
    #     old_feature_maps = old_data["feature_maps"]
    #     old_labels = old_data["labels"]

    np.savez(save_path, feature_maps=np.squeeze(feature_maps), labels=labels)
    print("SAVED")


get_feature_map("dataset/target", "dataset/feature_maps/target.npz")
