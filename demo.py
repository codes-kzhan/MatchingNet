import numpy as np
import tensorflow as tf
import matchnet


def main(support_set_path, target_path, model_dir):

    support_set = np.load(support_set_path)
    target = np.load(target_path)

    def tranfer(raw_data):
        raw_feature_maps = raw_data["feature_maps"]
        raw_labels = raw_data["labels"]

        shape = raw_feature_maps.shape

        feature_maps = np.reshape(raw_feature_maps, (-1, shape[2], shape[3], shape[4]))
        labels = np.reshape(np.array([[label] * shape[1] for label in raw_labels]), (-1))

        return feature_maps, labels, shape[0], shape[1]

    support_feature_maps, support_labels, possible_classes, shot = tranfer(support_set)
    target_feature_maps, target_labels, _, _ = tranfer(target)

    model = matchnet.MatchingNet(possible_classes=possible_classes, shot=shot,
                                 fce=True, batch_size=1)
    model.build()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    module_file = tf.train.latest_checkpoint(model_dir)
    if module_file is not None:
        print("Load checkpoint from ", module_file)
        restorer = tf.train.Saver()
        restorer.restore(sess, module_file)
        print("Done")

    loss = []
    acc = []
    for i in range(target_labels.shape[0]):
        feed_dict = {model.x_hat_encode: target_feature_maps[i],
                     model.y_hat_ind: target_labels[i],
                     model.x_i_encode: support_feature_maps,
                     model.y_i_ind: support_labels}

        cur_loss, cur_acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        loss.append(cur_loss)
        acc.append(cur_acc)
        print("loss: ", cur_loss, "acc: ", cur_acc)

    print("loss: ", np.mean(loss), "acc: ", np.mean(acc))

main("dataset/feature_maps/support_set.npz", "dataset/feature_maps/target.npz", "tmp/mobilenet_fce")
