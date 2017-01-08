from get_wjn_content import get_ksk_contents
from analysis_morphologic import create_datasets
import tensorflow as tf
import sys
import argparse

if __name__ == '__main__':

    # command_line_args
    parser = argparse.ArgumentParser()
    parser.add_argument("-o",
                        help="wjn article url",
                        type=str)
    parser.add_argument("-s",
                        help="String Mode")
    parser.add_argument("-a",
                        help="show article mode",
                        action="store_true")

    args = parser.parse_args()
    if args.o:
        url = [args.o]
        content_text = get_ksk_contents(url, single=True)
        contents = [i.text for i in content_text]
    elif args.s:
        contents = [args.s]
    else:
        raise ValueError

    dict = create_datasets(contents)
    dict_size = len(dict[0].tolist())

    # MLP params
    hidden1_size = 50
    hidden2_size = 10
    output_size = 2

    x = tf.placeholder(tf.float32, shape=[None, dict_size])

    with tf.name_scope("inference") as scope:
        W1 = tf.Variable(tf.zeros([dict_size, hidden1_size]),
                         name="weight_1")
        b1 = tf.Variable(tf.zeros([hidden1_size]),
                         name="bias_1")

        h1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

        W2 = tf.Variable(tf.zeros([hidden1_size, hidden2_size]),
                         name="weight_2")
        b2 = tf.Variable(tf.zeros([hidden2_size]),
                         name="bias_2")

        h2 = tf.nn.sigmoid(tf.matmul(h1, W2) + b2)

        W3 = tf.Variable(tf.zeros([hidden2_size, output_size]),
                         name="weight_3")
        b3 = tf.Variable(tf.zeros([output_size]),
                         name="bias_3")

    y = tf.nn.softmax(tf.nn.relu(tf.matmul(h2, W3) + b3))
    saver = tf.train.Saver([W1, b1, W2, b2, W3, b3])

    with tf.Session() as sess:
        saver.restore(sess, "wjn_model.ckpt")
        result = sess.run(y, feed_dict={x: dict})

    hantei_string = "\nウィーン　イク〜\n" \
                    "㌰㌰㌰㌰㌰㌰\n"\
                    "ﾀﾀﾞｲﾏｹｲｻﾝﾁｭｳﾃﾞｽ\n" \
                    "㌰㌰㌰㌰㌰㌰㌰\n" \
                    "ﾊﾝﾃｲｶﾞｶﾝﾘｮｳｼﾏｼﾀ\n"\
                    "ギャー!ギャー!ギャーッ!\n"
    if args.a:
        print("<記事>e\n%s" % contents[0])
    print(hantei_string)

    if result[0][0] > result[0][1]:
        print("\t柏木%.0fパーセント" % float(result[0][0]*100))
    else:
        print("\t奈倉%.0fパーセント" % float(result[0][1]*100))
