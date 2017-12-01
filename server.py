import tensorflow as tf
import numpy as numpy
import matplotlib.pyplot as pyplot
from flask import Flask, Response, request

app = Flask(__name__)

@app.route('/')
def page():
    with open('main.html', 'r', encoding='utf8') as f:
        return f.read()

sess = tf.InteractiveSession()

def setum_net():
    x = tf.placeholder(tf.float32, [None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d