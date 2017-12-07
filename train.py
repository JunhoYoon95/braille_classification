import tensorflow as tf
import numpy as np
import load_data
from datetime import datetime
import random
import matplotlib.pyplot as plt
import cv2

learning_rate = 0.001
training_epochs = 15
batch_size = 100


X = tf.placeholder(tf.float32)
X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28x28x1 (black/white)
Y = tf.placeholder(tf.float64, [None, 26])
training = tf.placeholder(tf.bool)

global_step = tf.Variable(0, trainable=False, name='global_step')

with tf.name_scope('layer1'):
    conv1 = tf.layers.conv2d(inputs=X_img, filters=32,
                             padding='SAME', kernel_size=[3, 3],
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                    padding="SAME", strides=2)

    dropout1 = tf.layers.dropout(inputs=pool1,
                                 rate=0.7, training=training)

with tf.name_scope('layer2'):
    conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3],
                             padding="SAME", activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                    padding="SAME", strides=2)

    dropout2 = tf.layers.dropout(inputs=pool2,
                                 rate=0.7, training=training)


with tf.name_scope('layer3'):
    conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3],
                             padding="same", activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())

    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
                                    padding="same", strides=2)

    dropout3 = tf.layers.dropout(inputs=pool3,
                                 rate=0.7, training=training)

with tf.name_scope('dense1'):
    flat = tf.reshape(dropout3, [-1, 128 * 4 * 4])

    dense4 = tf.layers.dense(inputs=flat,
                             units=625, activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())
    dropout4 = tf.layers.dropout(inputs=dense4,
                                 rate=0.5, training=training)

with tf.name_scope('optimizer'):
    logits = tf.layers.dense(inputs=dropout4, units=26)

    # define cost/loss & optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



# initialize
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs', sess.graph)

# train my model
print('Learning started. It takes sometime.')

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = 520

    for i in range(total_batch):
        batch_xs, batch_ys = load_data.get_braille()
        feed_dict = {X: batch_xs, Y: batch_ys, training: True}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
        print('Epoch:', '%04d of ' % (epoch + 1), '%04d' % (i + 1),  'cost =', '{:.9f}'.format(avg_cost))

    sess.run(global_step)
    summary = sess.run(merged, feed_dict={X: batch_xs, Y: batch_ys})
    writer.add_summary(summary, global_step=sess.run(global_step))

    file_name = './model/' + str(datetime.today().year) + '-' + str(datetime.today().month) + '-' + str(
        datetime.today().day) + '-' + str(datetime.today().hour) + '-tbd.ckpt'
    print(file_name)
    saver.save(sess, file_name, global_step=global_step)

    print()
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    print()

print('Learning Finished!')
file_name = './model/' + str(datetime.today().year) + '-' + str(datetime.today().month) + '-' + str(datetime.today().day) + '-' + str(datetime.today().hour) +'-tbd.ckpt'
print(file_name)
saver.save(sess, file_name, global_step=global_step)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: load_data.get_test_braille(get_image=True), Y: load_data.get_test_braille(get_label=True)}))


