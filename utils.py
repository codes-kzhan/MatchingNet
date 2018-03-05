import numpy as np
import tensorflow as tf
import os
from scipy import misc
import mobilenet_v1


class DataUtils:

    def __init__(self, data_path, data_type='IMAGE'):

        self.data_path = data_path
        self.data_type = data_type

        roots = []
        dirs_all = []
        files_all = []
        for root, dirs, files in os.walk(self.data_path):
            roots.append(root)
            dirs_all.append(dirs)
            files_all.append([os.path.join(root, file) for file in files])

        self.label_map = dirs_all[0]
        class_num = len(self.label_map)
        self.label = range(class_num)

        self.feature_path = files_all[1:(class_num + 1)]
        file_num = sum([len(file_paths) for file_paths in self.feature_path])

        print("Find " + str(class_num) + " classes")
        print("Find " + str(file_num) + " files")

    def get_batch(self, possible_classes=5, shot=5, batch_size=32, pic_size=224, channel=3):

        sample_num = possible_classes * shot
        support_feature = np.zeros((batch_size, sample_num, pic_size, pic_size, channel))
        support_label = np.zeros((batch_size, sample_num))
        target_feature = np.zeros((batch_size, pic_size, pic_size, channel))
        target_label = np.zeros(batch_size)

        for i in range(batch_size):
            class_index = np.random.choice(len(self.feature_path), possible_classes, False)
            target_class = np.random.randint(possible_classes)
            for j in range(possible_classes):
                files = self.feature_path[class_index[j]]

                if j == target_class:
                    file_index = np.random.choice(len(files), shot + 1, False)
                    file_index, target_file_index = np.split(file_index, [shot])
                    target_file_path = files[target_file_index[0]]

                    img = misc.imread(target_file_path)
                    img_resized = misc.imresize(img, [pic_size, pic_size])
                    img_resized = img_resized.reshape([pic_size, pic_size, channel])

                    target_feature[i, :, :, :] = img_resized
                    target_label[i] = target_class
                else:
                    file_index = np.random.choice(len(files), shot, False)

                for k in range(shot):
                    file_path = files[file_index[k]]
                    img = misc.imread(file_path)
                    img_resized = misc.imresize(img, [pic_size, pic_size])
                    img_resized = img_resized.reshape([pic_size, pic_size, channel])

                    support_feature[i, j * shot + k, :, :, :] = img_resized
                    support_label[i, j * shot + k] = j

        return support_feature, support_label, target_feature, target_label


def get_feature_map(data_path, save_path, pic_size=224, channel=3):

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

    features = tf.placeholder(tf.float32, shape=[None, pic_size, pic_size, channel])

    logits, _ = mobilenet_v1.mobilenet_v1(features, prediction_fn=None,
                                          reuse=tf.AUTO_REUSE, scope='FeatureExtractor/MobilenetV1')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    module_file = tf.train.latest_checkpoint("tmp/mobilenet_matchnet")
    print("Load checkpoint from ", module_file)
    restorer = tf.train.Saver()
    restorer.restore(sess, module_file)
    print("Done")

    feature_maps = []
    for paths in feature_path:
        imgs = []
        for path in paths:
            img = misc.imread(path)
            img_resized = misc.imresize(img, [pic_size, pic_size])
            img_resized = img_resized.reshape([pic_size, pic_size, channel])
            imgs.append(img_resized)
        feed_dict = {features: imgs}
        feature_map = sess.run([logits], feed_dict=feed_dict)
        feature_maps.append(feature_map)

    np.savez(save_path, feature_maps=feature_maps, labels=labels)


def get_minibatch(data, batch_size, shot, possible_class):

    pic_size = data.shape[2]
    n_samples = shot * possible_class
    mb_x_i = np.zeros((batch_size, n_samples, pic_size, pic_size, 1))
    mb_y_i = np.zeros((batch_size, n_samples))
    # mb_x_hat = np.zeros((batch_size, pic_size, pic_size, 1), dtype=np.int)
    # mb_y_hat = np.zeros((batch_size,), dtype=np.int)
    mb_x_hat = np.zeros((batch_size, pic_size, pic_size, 1))
    mb_y_hat = np.zeros((batch_size,))
    for i in range(batch_size):
        ind = 0
        pinds = np.random.permutation(n_samples)
        classes = np.random.choice(data.shape[0], possible_class, False)
        x_hat_class = np.random.randint(possible_class)
        for j,cur_class in enumerate(classes): #each class
            example_inds = np.random.choice(data.shape[1], shot, False)
            for eind in example_inds:
                mb_x_i[i, pinds[ind], :, :, 0] = np.rot90(data[cur_class][eind],np.random.randint(4))
                # mb_x_i[i, pinds[ind], :, :, 0] = data[cur_class][eind]
                mb_y_i[i, pinds[ind]] = j
                ind +=1
            if j == x_hat_class:
                mb_x_hat[i, :, :, 0] = np.rot90(data[cur_class][np.random.choice(data.shape[1])], np.random.randint(4))
                # mb_x_hat[i, :, :, 0] = data[cur_class][np.random.choice(data.shape[1])]
                mb_y_hat[i] = j
    return mb_x_i, mb_y_i, mb_x_hat, mb_y_hat


def get_numpy_data(filename, possible_classes):

    with np.load(filename) as data:
        features = data["features"]
        labels = data["labels"]

    assert features.shape[0] == labels.shape[0]

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    dataset = dataset.shuffle(2000).repeat().batch(possible_classes)

    return dataset.make_one_shot_iterator().get_next()


def get_batch(sess, dataset, batch_size, shot):

    support_features = []
    support_labels = []
    hat_features = []
    hat_labels = []

    for i in range(batch_size):
        raw_feature, raw_label = sess.run(dataset)

        hat_class = np.random.randint(raw_feature.shape[0])
        hat_labels.append(hat_class)

        feature = []
        label = []
        for j in range(raw_feature.shape[0]):
            if j == hat_class:
                index = np.random.choice(raw_feature.shape[1], shot + 1, False)
                hat_features.append(raw_feature[j][index[-1]])
            else:
                index = np.random.choice(raw_feature.shape[1], shot, False)

            feature.extend(raw_feature[j][index[:shot]])
            for k in range(shot):
                label.append(j)

        support_features.append(feature)
        support_labels.append(label)

    return np.array(support_features), np.array(support_labels), \
           np.array(hat_features), np.array(hat_labels)
