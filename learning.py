import wjn
import tensorflow as tf
import numpy as np
import os
import random

os.system('rm -rf ./data')
random.seed(0)
np.random.seed(0)
tf.set_random_seed(0)


if __name__ == '__main__':
    wjn = wjn.read_data_sets()
    dict_size = wjn.train.dict_size
    # print(dict_size)
    hidden1_size = 50
    hidden2_size = 10
    output_size = 2

    x = tf.placeholder(tf.float32, shape=[None, dict_size])
    y_ = tf.placeholder(tf.float32, shape=[None, output_size])
    keep_prob = tf.placeholder(tf.float32)

    with tf.name_scope("inference") as scope:
        W1 = tf.Variable(tf.truncated_normal([dict_size, hidden1_size]),
                         name="weight_1")
        b1 = tf.Variable(tf.truncated_normal([hidden1_size]),
                         name="bias_1")

        h1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

        h1_drop = tf.nn.dropout(h1, keep_prob)
        W2 = tf.Variable(tf.truncated_normal([hidden1_size, hidden2_size]),
                         name="weight_2")
        b2 = tf.Variable(tf.truncated_normal([hidden2_size]),
                         name="bias_2")

        h2 = tf.nn.sigmoid(tf.matmul(h1_drop, W2) + b2)
        h2_drop = tf.nn.dropout(h2, keep_prob)

        W3 = tf.Variable(tf.truncated_normal([hidden2_size, output_size]),
                         name="weight_3")
        b3 = tf.Variable(tf.truncated_normal([output_size]),
                         name="bias_3")

        y = tf.nn.softmax(tf.nn.relu(tf.matmul(h2_drop, W3) + b3))
        tf.histogram_summary('y[0]', y[0])

    with tf.name_scope("loss") as scope:
        cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
        tf.scalar_summary('entropy', cross_entropy)

    with tf.name_scope("training") as scope:
        train_step = tf.train.MomentumOptimizer(
            learning_rate=0.01, momentum=0.5).minimize(cross_entropy)

    with tf.Session() as sess:
        summary_writer = tf.train.SummaryWriter('./data', sess.graph)

        summary_op = tf.merge_all_summaries()

        sess.run(tf.initialize_all_variables())

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        test_dict = {x: wjn.test.dicts, y_: wjn.test.labels, keep_prob: 1.0}

        for i in range(30000):
            batch = wjn.train.next_batch(50)

            train_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5}
            train_step.run(feed_dict=train_dict)
            if i % 10 == 0:
                feed_dict = {x: batch[0], y_: batch[1], keep_prob: 1.0}
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                train_accuracy = accuracy.eval(feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, i)
                if i % 1000 == 0:
                    print("Accuracy(epoch %d): %f\nTrain Accuracy: %f" %
                          (i, accuracy.eval(feed_dict=test_dict),
                           train_accuracy))

        saver = tf.train.Saver([W1, b1, W2, b2, W3, b3])
        saver.save(sess, "wjn_model.ckpt")

        print(accuracy.eval(feed_dict=test_dict))
