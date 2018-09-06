'''
# @Time    : 18-9-6 下午2:02
# @Author  : ShengZ
# @FileName: lstm_mnist.py
# @Software: PyCharm
# @Github  : https://github.com/ZZshengyeah
'''

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/", one_hot=True)

learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

num_input = 28
time_steps = 28
num_hidden = 128
num_labels = 10

x = tf.placeholder('float',[None,time_steps,num_input])
y = tf.placeholder('float',[None,num_labels])

weights = tf.Variable(tf.random_normal([num_hidden,num_labels]))
biases = tf.Variable(tf.random_normal([num_labels]))

def LSTM(x, weights, biases):
    x = tf.unstack(x,time_steps,axis=1)
    lstm_cell = rnn.BasicLSTMCell(num_hidden,forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell,x,dtype=tf.float32)
    return tf.matmul(outputs[-1],weights) + biases

logits = LSTM(x, weights, biases)
prediction = tf.nn.softmax(logits)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)

correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as  sess:
    sess.run(init)
    for step in range(1,training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        #print('batch_x: {}'.format(batch_x.shape))
        #print('batch_y: {}'.format(batch_y.shape))
        batch_x = batch_x.reshape([batch_size,time_steps,num_input])
        #print('after batch_x: {}'.format(batch_x.shape))
        sess.run(train_op,feed_dict={x:batch_x,y:batch_y})
        if step % 50 == 0:
            losses, acc = sess.run([loss, accuracy], feed_dict={x:batch_x,y:batch_y})
            print('step: {}  loss: {}  acc: {}'.format(step,losses,acc))
    print('Optimization finished!')

    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, time_steps, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={x: test_data, y: test_label}))