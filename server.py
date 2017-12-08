import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pyplot
from flask import Flask, Response, request

app = Flask(__name__)

@app.route('/')
def page():
    with open('C:/Users/ML-10301/Documents/GitHub/braille_classification/index.html', 'r', encoding='utf8') as f:
        return f.read()

sess = tf.InteractiveSession()

def setup_net():

    X = tf.placeholder(tf.float32)
    X_img = tf.reshape(X, [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(inputs=X_img, filters=32,
                             padding='SAME', kernel_size=[3, 3],
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                    padding="SAME", strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3],
                             padding="SAME", activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                    padding="SAME", strides=2)

    conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3],
                             padding="same", activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())

    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
                                    padding="same", strides=2)

    flat = tf.reshape(pool3, [-1, 128 * 4 * 4])

    dense4 = tf.layers.dense(inputs=flat,
                             units=625, activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())

    logits = tf.layers.dense(inputs=dense4, units=26)

    result = tf.argmax(logits, 1)



    with tf.Session() as sess:
        # saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph('C:/Users/ML-10301/Documents/GitHub/braille_classification/logs/model-519.meta')
        saver.restore(sess, "C:/Users/ML-10301/Documents/GitHub/braille_classification/logs/model-519")
        print(saver.restore(sess, "C:/Users/ML-10301/Documents/GitHub/braille_classification/logs/model-519")
)
    return X, result

x, result = setup_net()

@app.route('/determine', methods=['POST'])
def determine():
    global x
    amts = [int(x, 16) / 255 for x in request.form['amounts'].split(',')]
    pyplot.imshow(np.array(amts).reshape((28, 28)))
    pyplot.show()
    res = sess.run(result, feed_dict={x: [amts]})
    return str(res[0])

if __name__ == '__main__':
    app.run(host='localhost', debug=True)