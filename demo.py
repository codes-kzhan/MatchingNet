import numpy as np
import tensorflow as tf
import matchnet


def main(support_set_path, target_path, model_dir):

    support_set = np.load(support_set_path)
    target = np.load(target_path)

    support_feature_maps = support_set["feature_maps"]
    support_label_map = support_set["labels"]
    target_feature_maps = target["feature_maps"]
    target_label_map = target["labels"]

    support_shape = support_feature_maps.shape
    target_shape = target_feature_maps.shape

    possible_classes = support_shape[0]
    shot = support_shape[1]
    batch_size = target_shape[0] * target_shape[1]

    support_feature_maps = [np.reshape(support_feature_maps, (-1, support_shape[2]))] * batch_size
    target_feature_maps = np.reshape(target_feature_maps, (-1, target_shape[2]))

    support_labels = []
    for i in range(possible_classes):
        support_labels.extend([i] * shot)
    support_labels = [support_labels] * batch_size

    target_labels = []
    for label in target_label_map:
        target_labels.extend(list(np.where(support_label_map == label)[0]) * target_shape[1])

    model = matchnet.MatchingNet(possible_classes=possible_classes, shot=shot,
                                 fce=True, batch_size=batch_size)
    model.build()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    module_file = tf.train.latest_checkpoint(model_dir)
    if module_file is not None:
        print("Load checkpoint from ", module_file)
        restorer = tf.train.Saver()
        restorer.restore(sess, module_file)
        print("Done")

    feed_dict = {model.x_hat_encode: target_feature_maps,
                 model.y_hat_ind: target_labels,
                 model.x_i_encode: support_feature_maps,
                 model.y_i_ind: support_labels}

    loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
    print("loss: ", loss, "acc: ", acc)


main("dataset/feature_maps/support_set.npz", "dataset/feature_maps/target.npz", "tmp/mobilenet_fce")
