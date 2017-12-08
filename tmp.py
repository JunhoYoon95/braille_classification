import tensorflow as tf
import random
# import matplotlib.pyplot as plt
import os
import load_data

tf.set_random_seed(777)  # reproducibility

# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

CHECK_POINT_DIR = TB_SUMMARY_DIR = './logs'


# input place holders
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32, [None, 26])
# Image input
X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28x28x1 (black/white)
tf.summary.image('input', X_img, 3)

# dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
training = tf.placeholder(tf.bool)

# weights & bias for nn layers
# http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
with tf.variable_scope('layer1'):
    conv1 = tf.layers.conv2d(inputs=X_img, filters=32,
                             padding='SAME', kernel_size=[3, 3],
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                    padding="SAME", strides=2)

    dropout1 = tf.layers.dropout(inputs=pool1,
                                 rate=0.7, training=training)

with tf.variable_scope('layer2'):
    conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3],
                             padding="SAME", activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                    padding="SAME", strides=2)

    dropout2 = tf.layers.dropout(inputs=pool2,
                                 rate=0.7, training=training)

with tf.variable_scope('layer3'):
    conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3],
                             padding="same", activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())

    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
                                    padding="same", strides=2)

    dropout3 = tf.layers.dropout(inputs=pool3,
                                 rate=0.7, training=training)

with tf.variable_scope('layer4'):
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


# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

tf.summary.scalar("loss", cost)

last_epoch = tf.Variable(0, name='last_epoch')

# Summary
summary = tf.summary.merge_all()

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Create summary writer
writer = tf.summary.FileWriter(TB_SUMMARY_DIR)
writer.add_graph(sess.graph)
global_step = 0

# Saver and Restore
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECK_POINT_DIR)

if checkpoint and checkpoint.model_checkpoint_path:
    try:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    except:
        print("Error on loading old network weights")
else:
    print("Could not find old network weights")

start_from = sess.run(last_epoch)

# train my model
print('Start learning from:', start_from)

for epoch in range(start_from, training_epochs):
    print('Start Epoch:', epoch)

    avg_cost = 0
    total_batch = 520

    for i in range(total_batch):
        batch_xs, batch_ys = load_data.get_braille()
        feed_dict = {X: batch_xs, Y: batch_ys, training: True}
        s, _ = sess.run([summary, optimizer], feed_dict=feed_dict)
        writer.add_summary(s, global_step=global_step)
        global_step += 1

        avg_cost += sess.run(cost, feed_dict=feed_dict) / total_batch
        print('Epoch:', '%04d of ' % (epoch + 1), '%04d' % (i + 1), 'cost =', '{:.9f}'.format(avg_cost))


    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print("Saving network...")
    sess.run(last_epoch.assign(epoch + 1))
    if not os.path.exists(CHECK_POINT_DIR):
        os.makedirs(CHECK_POINT_DIR)
    saver.save(sess, CHECK_POINT_DIR + "/model", global_step=i)

print('Learning Finished!')