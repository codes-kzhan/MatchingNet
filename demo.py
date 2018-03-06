import tensorflow as tf
import matchnet as mn
import utils
import os


def train(model_dir, train_data, val_data=None, possible_classes=5, shot=5,
          batch_size=32, learning_rate=1e-3, val_gap=50):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model = mn.MatchingNet(possible_classes=possible_classes, shot=shot, fce=True,
                           batch_size=batch_size)
    model.build()

    optim = tf.train.AdamOptimizer(learning_rate)
    train_step = optim.minimize(model.loss)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    module_file = tf.train.latest_checkpoint(model_dir)
    if module_file is not None:
        print("Load checkpoint from ", module_file)
        restorer = tf.train.Saver()
        restorer.restore(sess, module_file)
        print("Done")

    dataset_train = utils.DataUtils(train_data)
    # dataset_val = utils.DataUtils(val_data)

    saver = tf.train.Saver(max_to_keep=3)

    for i in range(int(1e7)):
        mb_x_i_encode, mb_y_i, mb_x_hat_encode, mb_y_hat = dataset_train.get_batch(possible_classes=possible_classes,
                                                                     shot=shot, batch_size=batch_size)

        feed_dict = {model.x_hat_encode: mb_x_hat_encode,
                     model.y_hat_ind: mb_y_hat,
                     model.x_i_encode: mb_x_i_encode,
                     model.y_i_ind: mb_y_i}

        _, loss, acc = sess.run([train_step, model.loss, model.acc], feed_dict=feed_dict)

        print(i, "loss: ", loss, "acc: ", acc)

        if i % val_gap == 0 and not i == 0:

            # mb_x_i, mb_y_i, mb_x_hat, mb_y_hat = dataset_val.get_batch(possible_classes=possible_classes,
            #                                                          shot=shot, batch_size=batch_size)
            # feed_dict = {model.x_hat_encode: mb_x_hat,
            #              model.y_hat_ind: mb_y_hat,
            #              model.x_i_encode: mb_x_i,
            #              model.y_i_ind: mb_y_i}
            # loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)

            global_step += i
            saver.save(sess, "tmp/mobilenet_fce/55", global_step=global_step)


def evaluate(data_path):

    model = mn.MatchingNet()
    model.build()

    dataset = utils.DataUtils(data_path)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    mb_x_i_encode, mb_y_i, mb_x_hat_encode, mb_y_hat = dataset.get_batch(batch_size=1000)

    feed_dict = {model.x_hat_encode: mb_x_hat_encode,
                 model.y_hat_ind: mb_y_hat,
                 model.x_i_encode: mb_x_i_encode,
                 model.y_i_ind: mb_y_i}
    acc = sess.run([model.acc], feed_dict=feed_dict)

    print("acc: ", acc)


# evaluate("dataset/feature_maps/feature_map_mobilenet.npz")
train("tmp/mobilenet_fce", "dataset/feature_maps/val.npz")
