# Importing all the dependencies
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Defining the dataset
mnist = input_data.read_data_sets("MNIST_data/")

n_input = 28 * 28  # As the dimensions of all the input images are 28 * 28
n_hidden1 = 500
n_hidden2 = 500
n_hidden3 = 20  # Codings
n_hidden4 = n_hidden2
n_hidden5 = n_hidden1
n_output = n_input

# activation_fn=tf.nn.elu #Using 'ELU' as the activation function here
# weights_initializer=tf.contrib.layers.variance_scaling_initializer()

x = tf.placeholder(tf.float32, shape=[None, n_input])

# 1st Hidden Layer
stddev1 = 2 / np.sqrt(n_input)
q1 = tf.truncated_normal([n_input, n_hidden1], stddev=stddev1)
w1 = tf.Variable(q1)
b1 = tf.Variable(tf.zeros([n_hidden1]))
hidden1 = tf.nn.elu(tf.matmul(x, w1) + b1)

# 2nd Hidden Layer
stddev2 = 2 / np.sqrt(n_hidden1)
q2 = tf.truncated_normal([n_hidden1, n_hidden2], stddev=stddev2)
w2 = tf.Variable(q2)
b2 = tf.Variable(tf.zeros([n_hidden2]))
hidden2 = tf.nn.elu(tf.matmul(hidden1, w2) + b2)

# 3rd Hidden Layer
stddev3 = 2 / np.sqrt(n_hidden2)
q3 = tf.truncated_normal([n_hidden2, n_hidden3], stddev=stddev3)
w3 = tf.Variable(q3)
b3 = tf.Variable(tf.zeros([n_hidden3]))
hidden3_mean = tf.matmul(hidden2, w3) + b3
hidden3_gamma = tf.matmul(hidden2, w3) + b3
hidden3_sigma = tf.exp(0.5 * hidden3_gamma)
noise = tf.random_normal(tf.shape(hidden3_sigma), dtype=tf.float32)
hidden3 = hidden3_mean + hidden3_sigma * noise

# 4th Hidden Layer
stddev4 = 2 / np.sqrt(n_hidden3)
q4 = tf.truncated_normal([n_hidden3, n_hidden4], stddev=stddev4)
w4 = tf.Variable(q4)
b4 = tf.Variable(tf.zeros([n_hidden4]))
hidden4 = tf.nn.elu(tf.matmul(hidden3, w4) + b4)

# 5th Hidden Layer
stddev5 = 2 / np.sqrt(n_hidden4)
q5 = tf.truncated_normal([n_hidden4, n_hidden5], stddev=stddev5)
w5 = tf.Variable(q5)
b5 = tf.Variable(tf.zeros([n_hidden5]))
hidden5 = tf.nn.elu(tf.matmul(hidden4, w5) + b5)

# Output Layer
stddev6 = 2 / np.sqrt(n_hidden5)
q6 = tf.truncated_normal([n_hidden5, n_output], stddev=stddev6)
w6 = tf.Variable(q6)
b6 = tf.Variable(tf.zeros([n_output]))
logits = tf.matmul(hidden5, w6) + b6
output = tf.sigmoid(logits)

# The Cost Function
reconstruction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=logits))
latent_loss = 0.5 * tf.reduce_sum(tf.exp(hidden3_gamma) + tf.square(hidden3_mean) - 1 - hidden3_gamma)
cost = reconstruction_loss + latent_loss

# Optimizer
learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(cost)

init = tf.global_variables_initializer()

n_digits = 50
n_epochs = 400  # The higher the value of 'n_epochs' the better the results
batch_size = 150


def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")


# Starting the Session
with tf.Session() as sess:
    init.run()  # Initializing all the global variables
    print("The model will start generating results only after 400 epochs are completed...")
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={x: x_batch})
        print("Epoch ", epoch + 1, " is complete")

    codings_rnd = np.random.normal(size=[n_digits, n_hidden3])
    outputs_val = output.eval(feed_dict={hidden3: codings_rnd})

    # Generating the digits after the model has been trained
    for iteration in range(n_digits):
        plt.subplot(math.ceil(n_digits / 10), 10, iteration + 1)  # subplot(n_rows, n_cols) is used to plot all the plots in the same window
        plot_image(outputs_val[iteration])
    plt.show()
