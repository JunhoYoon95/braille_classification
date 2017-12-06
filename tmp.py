from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

_x, _y=  mnist.train.next_batch(100)
print(type(_y))
print(_y.shape)
print(_y[0])