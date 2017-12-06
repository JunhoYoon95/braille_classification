import tensorflow as tf
import numpy as np
import load_data
import random
import matplotlib.pyplot as plt
import cv2

learning_rate = 0.001
training_epochs = 15
batch_size = 100


X = tf.placeholder(tf.float32)
X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28x28x1 (black/white)
Y = tf.placeholder(tf.string, [None, 26])
print(Y)
training = tf.placeholder(tf.bool)

conv1 = tf.layers.conv2d(inputs=X_img, filters=32,
                            padding='SAME', kernel_size=[3, 3],
                            activation=tf.nn.relu)

pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                padding="SAME", strides=2)

dropout1 = tf.layers.dropout(inputs=pool1,
                             rate=0.7, training=training)


conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3],
                        padding="SAME", activation=tf.nn.relu)

pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                padding="SAME", strides=2)

dropout2 = tf.layers.dropout(inputs=pool2,
                            rate=0.7, training=training)


conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3],
                        padding="same", activation=tf.nn.relu)

pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
                                padding="same", strides=2)

dropout3 = tf.layers.dropout(inputs=pool3,
                             rate=0.7, training=training)

flat = tf.reshape(dropout3, [-1, 128 * 4 * 4])

dense4 = tf.layers.dense(inputs=flat,
                        units=625, activation=tf.nn.relu)

logits = tf.layers.dense(inputs=dense4, units=26)

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
print('Learning started. It takes sometime.')
# batch_xs, batch_ys = load_data.get_braille()
batch_xs = cv2.imread("Braille/character12/0203_15.png", 0)
batch_ys = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
print(batch_ys)
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = 520

    for i in range(total_batch):
        feed_dict = {X: batch_xs, Y: batch_ys, training: True}
        # plt.imshow(batch_xs[284],cmap="gray")
        # plt.show()

        # print(sess.run([cost, optimizer], feed_dict=feed_dict))
        x = sess.run([cost, optimizer], feed_dict=feed_dict)
        # avg_cost += c / total_batch
        print(x)

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy

# if you have a OOM error, please refer to lab-11-X-mnist_deep_cnn_low_memory.py

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: load_data.get_test_braille(get_image=True), Y: load_data.get_test_braille(get_label=True), keep_prob: 1}))
#
# # Get one and predict
# r = random.randint(0, load_data.get_braille(get_size=True))
# print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
# print("Prediction: ", sess.run(
#     tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1}))
