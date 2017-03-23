import cPickle, gzip
import numpy as np
import tensorflow as tf
import tflearn


def relu(x, leakiness=0.1):
    return tf.select(tf.less(x, 0.0), leakiness * x, x)


def full_highway(x, fan_out):

    # Normal layer
    H = tflearn.fully_connected(x, fan_out, activation='linear', bias=True,
                                weights_init=tflearn.initializations.xavier(),
                                bias_init='uniform', regularizer='L2', weight_decay=0.0001)

    # Transform gate
    T = tflearn.fully_connected(x, fan_out, activation='linear', bias=True,
                                weights_init=tflearn.initializations.xavier(),
                                bias_init='uniform', regularizer='L2', weight_decay=0.0001)
    T = tf.nn.sigmoid(T)

    # Carry gate
    C = 1.0 - T
    res = relu(H * T)

    return (res + (C * x)), T

batch_size = 128

graph = tf.Graph()
with graph.as_default():

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    tc_multiplier = tf.placeholder(tf.float32)

    net = tflearn.fully_connected(x, 300, activation='linear', weights_init=tflearn.initializations.xavier(),
                                  bias_init='uniform', regularizer='L2', weight_decay=0.0001)
    net, t1 = full_highway(net, 300)
    net, t2 = full_highway(net, 300)
    net, t3 = full_highway(net, 300)

    net = tflearn.fully_connected(net, 10, activation='linear', weights_init=tflearn.initializations.xavier(),
                                  bias_init='uniform', regularizer='L2', weight_decay=0.0001)

    transform_sum = tf.reduce_sum(t1, axis=[1]) + tf.reduce_sum(t2, axis=[1]) + tf.reduce_sum(t3, axis=[1])

    # Calculate loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(net, y)
    loss = tf.reduce_mean(cross_entropy) + tc_multiplier * tf.reduce_mean(transform_sum)

    # Find all the correctly classified examples
    correct_ = tf.equal(tf.argmax(y, 1), tf.argmax(net, 1))

    # Optimizer with gradient clipping
    global_step = tf.Variable(0)
    lr = tf.placeholder(tf.float32)
    optimizer = tf.train.MomentumOptimizer(lr, 0.7)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1)
    optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

    # Op to initialize variables
    init_op = tf.global_variables_initializer()

# Load MNIST data
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
del test_set
train_x, train_y = train_set
valid_x, valid_y = valid_set
train_y = np.eye(10)[train_y]
valid_y = np.eye(10)[valid_y]

epochs = 100

losses = []
transforms_layer_1 = []; transforms_layer_2 = []; transforms_layer_3 = []
valid_transforms_layer_1 = []; valid_transforms_layer_2 = []; valid_transforms_layer_3 = []
valid_accuracies = []
learn_rate = 0.1
with tf.Session(graph=graph) as session:

    session.run(init_op)
    saver = tf.train.Saver()
    save_path = saver.save(session,'./initial-model')

    sequence = np.random.choice(len(train_x), size=len(train_x), replace=False)  # The sequence to form batches
    random_train_x = train_x[sequence]
    random_train_y = train_y[sequence]

    i = 1
    cursor = 0

    print "GETTING TRANSFORMATIONS FOR TEST SET TO CHECK SHAPES"
    for iii in xrange(100):
        batch_xs = valid_x[iii * 100: (iii + 1) * 100]
        batch_ys = valid_y[iii * 100: (iii + 1) * 100]
        feed_dict = {x: batch_xs, y: batch_ys}
        st1, st2, st3 = session.run([t1, t2, t3], feed_dict=feed_dict)

        valid_transforms_layer_1.extend(st1); valid_transforms_layer_2.extend(st2); valid_transforms_layer_3.extend(st3)
    print len(valid_transforms_layer_1), len(valid_transforms_layer_2), len(valid_transforms_layer_3)
    print valid_transforms_layer_1[0].shape, valid_transforms_layer_2[0].shape, valid_transforms_layer_3[0].shape
    valid_transforms_layer_1 = []; valid_transforms_layer_2 = []; valid_transforms_layer_3 = []

    while i <= epochs:

        batch_xs = random_train_x[cursor: min((cursor + batch_size), len(train_x))]
        batch_ys = random_train_y[cursor: min((cursor + batch_size), len(train_x))]
        if i < 21:
            feed_dict = {x: batch_xs, y: batch_ys, lr: learn_rate, tc_multiplier: 0.0}
        else:
            feed_dict = {x: batch_xs, y: batch_ys, lr: learn_rate, tc_multiplier: 0.05}

        # Train it on the batch
        tflearn.is_training(True, session=session)
        _, train_step = session.run([optimizer, global_step], feed_dict=feed_dict)

        if train_step < 20000:
            learn_rate = 0.1
        elif train_step < 30000:
            learn_rate = 0.01
        elif train_step < 40000:
            learn_rate = 0.001
        else:
            learn_rate = 0.0001

        cursor += batch_size
        if cursor > len(train_x):
            cursor = 0
            tflearn.is_training(False, session=session)
            l_list = []
            ac_list = []
            print "GETTING LOSSES FOR ALL EXAMPLES"
            for iii in xrange(500):
                batch_xs = train_x[iii * 100: (iii + 1) * 100]
                batch_ys = train_y[iii * 100: (iii + 1) * 100]
                feed_dict = {x: batch_xs, y: batch_ys}
                cr = session.run([cross_entropy], feed_dict=feed_dict)
                cr = cr[0]

                # Append losses, activations for batch
                l_list.extend(cr)
            # Append losses, activations for epoch
            losses.append(l_list)

            # Validation test
            print "TESTING ON VALID SET for epoch = " + str(i)
            cor_pred = []
            for iii in xrange(100):
                a = session.run([correct_], feed_dict={x: valid_x[iii * 100:(iii + 1) * 100],
                                                       y: valid_y[iii * 100:(iii + 1) * 100]})
                cor_pred.append(a)
            print "Accuracy = " + str(np.mean(cor_pred))
            valid_accuracies.append(np.mean(cor_pred))
            tflearn.is_training(True, session=session)
            sequence = np.random.choice(len(train_x), size=len(train_x), replace=False)  # The sequence to form batches
            random_train_x = train_x[sequence]
            random_train_y = train_y[sequence]
            i += 1

    tflearn.is_training(False, session=session)
    save_path = saver.save(session, './final-model')
    print "GETTING TRANSFORMATIONS FOR TRAIN SET"
    for iii in xrange(500):
        batch_xs = train_x[iii * 100: (iii + 1) * 100]
        batch_ys = train_y[iii * 100: (iii + 1) * 100]
        feed_dict = {x: batch_xs, y: batch_ys}
        st1, st2, st3 = session.run([t1, t2, t3], feed_dict=feed_dict)

        transforms_layer_1.extend(st1); transforms_layer_2.extend(st2); transforms_layer_3.extend(st3);

    print "GETTING TRANSFORMATIONS FOR TEST SET"
    for iii in xrange(100):
        batch_xs = valid_x[iii * 100: (iii + 1) * 100]
        batch_ys = valid_y[iii * 100: (iii + 1) * 100]
        feed_dict = {x: batch_xs, y: batch_ys}
        st1, st2, st3 = session.run([t1, t2, t3], feed_dict=feed_dict)

        valid_transforms_layer_1.extend(st1); valid_transforms_layer_2.extend(st2); valid_transforms_layer_3.extend(st3)
    np.save('transforms_layer_1', transforms_layer_1)
    np.save('transforms_layer_2', transforms_layer_2)
    np.save('transforms_layer_3', transforms_layer_3)
    np.save('valid_transforms_layer_1', valid_transforms_layer_1)
    np.save('valid_transforms_layer_2', valid_transforms_layer_2)
    np.save('valid_transforms_layer_3', valid_transforms_layer_3)
    np.save('losses', losses)
    np.save('valid_accuracies', valid_accuracies)