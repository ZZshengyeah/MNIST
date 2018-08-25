from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

input_node = 784
output_node = 10

layer1_node = 500
batch_size = 100

learning_rate_base = 0.8
learning_rate_decay = 0.99

regularization_rate = 0.0001
training_steps = 30000

moving_average_decay = 0.99


def forward(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    if avg_class == None:
        hidden_layer_input = tf.matmul(input_tensor,weights1) + biases1
        hidden_layer_output = tf.nn.relu(hidden_layer_input)
        return tf.matmul(hidden_layer_output,weights2) + biases2
    else:
        hidden_layer_input = tf.matmul(input_tensor,avg_class.average(weights1)) + avg_class.average(biases1)
        hidden_layer_output = tf.nn.relu(hidden_layer_input)
        return tf.matmul(hidden_layer_output,avg_class.average(weights2)) + avg_class.average(biases2)


def train(mnist):
    x = tf.placeholder(tf.float32,[None,input_node],name='x-input')
    y_ = tf.placeholder(tf.float32,[None,output_node],name='y-input')

    weights1 = tf.Variable(tf.truncated_normal([input_node,layer1_node],stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1,shape=[layer1_node]))

    weights2 = tf.Variable(tf.truncated_normal([layer1_node,output_node],stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1,shape=[output_node]))

    y=forward(x,None,weights1,biases1,weights2,biases2)
    global_step = tf.Variable(0,trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay,global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    average_y = forward(x,variable_averages,weights1,biases1,weights2,biases2)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
    regularization = regularizer(weights1) + regularizer(weights2)

    loss = cross_entropy + regularization

    learning_rate = tf.train.exponential_decay(learning_rate_base,global_step,mnist.train.num_examples,
                                               learning_rate_decay)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    train_op = tf.group(train_step,variables_averages_op)

    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
        test_feed = {x:mnist.test.images,y_:mnist.test.labels}

        for i in range(training_steps):
            if i%1000 == 0:
                validate_acc = sess.run(accuracy,feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy on average model is %g" % (i,validate_acc))
            xs,ys = mnist.train.next_batch(batch_size)
            sess.run(train_op,feed_dict={x:xs,y_:ys})

        test_acc = sess.run(accuracy,feed_dict=test_feed)
        print("After %d training step(s), test accuracy on average model is %g" % (training_steps,test_acc))

def main(argv=None):
    mnist = input_data.read_data_sets('./tmp/data',one_hot=True)
    train(mnist)

if __name__ == "__main__":
    tf.app.run()
