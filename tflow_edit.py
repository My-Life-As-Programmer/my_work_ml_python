import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
#  xw+b 
n_classes = 10
batch_size = 100

x = tf.placeholder('float',[None, 784]) 
y = tf.placeholder('float')

# (input_data * weights) + biases
def nn_model(data):
    h1_layer = {'weights' :tf.Variable(tf.random_normal([784, n_nodes_hl1])),'biases' :tf.Variable(tf.random_normal([n_nodes_hl1]))}
    h2_layer = {'weights' :tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),'biases' :tf.Variable(tf.random_normal([n_nodes_hl2]))}
    h3_layer = {'weights' :tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),'biases' :tf.Variable(tf.random_normal([n_nodes_hl3]))}
    op_layer = {'weights' :tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),'biases' :tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, h1_layer['weights']) , h1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, h2_layer['weights']) , h2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, h3_layer['weights']) , h3_layer['biases'])
    l3 = tf.nn.relu(l3)

    op = tf.matmul(l3, op_layer['weights']) + op_layer['biases']

    return op

def train_nn(x):
    prediction = nn_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    epochs = 10

    with tf.Session() as sess:
        #sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())﻿

        for epoch in range(epochs):
            ep_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                ep_x, ep_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x: ep_x, y: ep_y})
                ep_loss += c
                print('Epoch', epoch, ' completed out of ', epochs,' loss:',ep_loss)

            correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct,'float'))
            print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
                
train_nn(x)
   
   
   
 
