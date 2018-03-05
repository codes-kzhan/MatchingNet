import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import matchnet as mn
import utils
import os


def evaluate(model_dir):

    shot = 5
    possible_classes = 5
    model = mn.MatchingNet(28, shot, possible_classes, fce=False, channel=1)
    model.build()

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    module_file = tf.train.latest_checkpoint(model_dir)

    if module_file is not None:
        saver = tf.train.Saver()
        saver.restore(sess, module_file)

    dataset = utils.DataUtils("dataset/omni")

    mb_x_i, mb_y_i, mb_x_hat, mb_y_hat = dataset.get_batch(batch_size=500, pic_size=28, channel=1)
    feed_dict = {model.x_hat: mb_x_hat,
                 model.y_hat_ind: mb_y_hat,
                 model.x_i: mb_x_i,
                 model.y_i_ind: mb_y_i}

    res = sess.run([model.acc], feed_dict=feed_dict)

    print(res)


def train():

    shot = 5
    possible_classes = 5
    pic_size = 28

    load_dir = 'tmp/mnist_fce_sorted' + str(shot) + '_' + str(possible_classes)
    save_dir = 'tmp/mnist_fce_sorted' + str(shot) + '_' + str(possible_classes)
    save_path = save_dir + '/omniglot'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    learning_rate = 1e-5
    batch_size = 32
    test_gap = 500

    model = mn.MatchingNet(pic_size, shot, possible_classes, fce=True, channel=1)
    model.build()

    var = tf.trainable_variables()
    train_var = [val for val in var if 'lstm' in val.name]

    optim = tf.train.AdamOptimizer(learning_rate)
    grads = tf.gradients(model.loss, train_var)
    grads = list(zip(grads, train_var))
    train_step = optim.apply_gradients(grads_and_vars=grads)
    # train_step = optim.minimize(model.loss)

    dataset_train = utils.DataUtils("")
    dataset_test = utils.DataUtils("")

    var = tf.global_variables()
    global_step = [val for val in var if 'global_step' in val.name]
    if len(global_step):
        global_step = global_step[0]
    else:
        global_step = tf.Variable(0, name='global_step', trainable=False)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True
    sess = tf.Session(config=sess_config)

    sess.run(tf.global_variables_initializer())

    module_file = tf.train.latest_checkpoint(load_dir)

    if module_file is not None:
        print("Load checkpoint from ", module_file)
        # reader = pywrap_tensorflow.NewCheckpointReader(module_file)
        # var_name = list(reader.get_variable_to_shape_map().keys())
        # var_name = [val for val in var_name if 'global_step' not in val]
        # var = tf.global_variables()
        # var_to_restore = [val for val in var if val.name.strip(':0') in var_name]
        # print(var_to_restore)
        restorer = tf.train.Saver()  # var_to_restore)
        restorer.restore(sess, module_file)
        print("Done")

    saver = tf.train.Saver(max_to_keep=3)

    for i in range(int(1e7)):
        mb_x_i, mb_y_i, mb_x_hat, mb_y_hat = dataset_train.get_batch(possible_classes, shot)

        feed_dict = {model.x_hat: mb_x_hat,
                     model.y_hat_ind: mb_y_hat,
                     model.x_i: mb_x_i,
                     model.y_i_ind: mb_y_i}
        _, loss, acc = sess.run([train_step, model.loss, model.acc], feed_dict=feed_dict)
        print(i, "loss: ", loss, "acc: ", acc)

        if i % test_gap == 0 and not i == 0:

            mb_x_i, mb_y_i, mb_x_hat, mb_y_hat = dataset_test.get_batch(possible_classes, shot)
            feed_dict = {model.x_hat: mb_x_hat,
                         model.y_hat_ind: mb_y_hat,
                         model.x_i: mb_x_i,
                         model.y_i_ind: mb_y_i}
            loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
            print("val_loss: ", loss, "val_acc: ", acc)

            global_step += i
            saver.save(sess, save_path, global_step=global_step)


def mobile_eval():

    shot = 5
    possible_classes = 5
    pic_size = 224

    load_dir = "tmp/ssd_mobilenet_v1_coco_2017_11_17"

    model = mn.MatchingNet(pic_size, shot, possible_classes, fce=False, model_type='MOBILENET')
    model.build()

    dataset = utils.DataUtils("dataset/train")

    sess_config = tf.ConfigProto()
    sess_config.allow_soft_placement = True
    sess = tf.Session(config=sess_config)

    sess.run(tf.global_variables_initializer())

    module_file = tf.train.latest_checkpoint(load_dir)
    # module_file = 'tmp/mobilenet_v1_1.0_224_2017_06_14/mobilenet_v1_1.0_224.ckpt'
    if module_file is not None:
        print("Load checkpoint from ", module_file)
        reader = pywrap_tensorflow.NewCheckpointReader(module_file)
        var_name = list(reader.get_variable_to_shape_map().keys())
        # var_name = [val for val in var_name if 'global_step' not in val]
        var = tf.global_variables()
        var_to_restore = [val for val in var if val.name.strip(':0') in var_name]
        # print(var_to_restore)
        restorer = tf.train.Saver(var_to_restore)
        restorer.restore(sess, module_file)
        print("Done")

    # mb_x_i, mb_y_i, mb_x_hat, mb_y_hat = dataset.get_batch(batch_size=25)
    #
    # feed_dict = {model.x_hat: mb_x_hat,
    #              model.y_hat_ind: mb_y_hat,
    #              model.x_i: mb_x_i,
    #              model.y_i_ind: mb_y_i}
    # acc = sess.run([model.acc], feed_dict=feed_dict)
    # print("acc: ", acc)


mobile_eval()
# evaluate("tmp/mnist_convnet_model")