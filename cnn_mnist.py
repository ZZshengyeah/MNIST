'''
# @Time    : 18-8-25 上午9:46
# @Author  : ShengZ
# @FileName: cnn_mnist.py
# @Software: PyCharm
# @Github  : https://github.com/ZZshengyeah
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

input_node = 784
output_node = 10

image_size = 28
num_channels = 1
num_labels = 10

conv1_deep = 32
conv1_size = 5
conv2_deep = 64
conv2_size = 5

full_connect_node = 512

def cnn_forward(input_tensor,regularizer):
    with tf.variable_scope('layer1_conv1'):
        #卷积层权重： 卷积核长,卷积核宽,通道数,卷积核深度
        #              3/5     3/5        人为设置
        conv1_weights = tf.get_variable('weight',[conv1_size,conv1_size,num_channels,conv1_deep],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))

        conv1_biases = tf.get_variable('bias',[conv1_deep],initializer=tf.constant_initializer(0.0))
        input_tensor = tf.reshape(input_tensor,[-1,image_size,image_size,num_channels])
        conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')
        relu1 = tf.nn.relu(tf.add(conv1,conv1_biases))

    with tf.name_scope('layer2_pool1'):
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    with tf.variable_scope('layer3_conv2'):
        conv2_weights = tf.get_variable('weight',[conv2_size,conv2_size,conv1_deep,conv2_deep],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('bias',[conv2_deep],initializer=tf.constant_initializer(0.0))

        conv2 = tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding='SAME')
        relu2 = tf.nn.relu(tf.add(conv2,conv2_biases))

    with tf.name_scope('layer4_pool2'):
        pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshape = tf.reshape(pool2,[-1,nodes])

    with tf.variable_scope('layer5_full_connect1'):
        fc1_weight = tf.get_variable('weight',[nodes,full_connect_node],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc1_biases = tf.get_variable('bias',[full_connect_node],
                                     initializer=tf.constant_initializer(0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc1_weight))
        fc1 = tf.nn.sigmoid(tf.matmul(reshape,fc1_weight) + fc1_biases)
        fc1 = tf.nn.dropout(fc1,0.8)

    with tf.variable_scope('layer6_full_connect2'):
        fc2_weight = tf.get_variable('weight',[full_connect_node,output_node],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_biases = tf.get_variable('bias',[output_node],
                                     initializer=tf.constant_initializer(0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc2_weight))
        fc2 = tf.nn.sigmoid(tf.matmul(fc1,fc2_weight) + fc2_biases)

    return fc2

batch_size = 100
learning_rate_base = 0.8
learning_rate_decay = 0.99
regularaztion_rate = 0.0001
training_steps = 10000

def train(mnist):
    x = tf.placeholder(tf.float32,[None,input_node],
                       name='x-input')
    y_ = tf.placeholder(tf.float32,[None,output_node],name='y-input')

    validation_feed = {x: mnist.validation.images,
                       y_: mnist.validation.labels}
    test_feed = {x: mnist.test.images,
                 y_: mnist.test.labels}

    regularizer = tf.contrib.layers.l2_regularizer(regularaztion_rate)

    y = cnn_forward(x,regularizer)

    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    global_step = tf.Variable(0,trainable=False)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(learning_rate_base,
                                               global_step,
                                               mnist.train.num_examples / batch_size,
                                               learning_rate_decay)
    train_steps = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(int(mnist.train.num_examples / batch_size)):
            xs,ys = mnist.train.next_batch(batch_size)
            sess.run(train_steps,feed_dict={x:xs,y_:ys})
            validation_acc = sess.run(accuracy,feed_dict=validation_feed)
            if i % 1000 == 0:
                print("After %d training step(s), validation accuracy on average model is %g" % (i, validation_acc))

        test_acc = sess.run(accuracy,feed_dict=test_feed)
        print("After %d training step(s), test accuracy on average model is %g" % (training_steps,test_acc))

def main(argv=None):
    mnist = input_data.read_data_sets('./data',one_hot=True)
    train(mnist)

if __name__ == "__main__":
    tf.app.run()



