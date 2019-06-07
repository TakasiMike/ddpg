import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Κόμβοι σε κάθε hidden layer

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10  # Αριθμός nodes στο output
batch_size = 100

x = tf.placeholder('float', [None, 784])  # input
y = tf.placeholder('float')   # output


def neural_network_model(data):   # Ορισμός του νευρωνικού δικτύου

    # Ορισμός του μεγέθους του κάθε hidden layer, δηλαδή των weights & biases

    hidden_1_layer = {'weights': tf.Variable(tf.random.normal([784, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal[n_nodes_hl1])}

    hidden_2_layer = {'weights': tf.Variable(tf.random.normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal[n_nodes_hl2])}

    hidden_3_layer = {'weights': tf.Variable(tf.random.normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal[n_nodes_hl3])}

    output_layer = {'weights': tf.Variable(tf.random.normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal[n_classes])}

    # Μοντέλο του νευρωνικού δικτύου
    layer_1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])  # y = x * w + b
    layer_1 = tf.nn.relu(layer_1)   # activation function

    layer_2 = tf.add(tf.matmul(layer_1, hidden_2_layer['weights']), hidden_2_layer['biases'])  # y = x * w + b
    layer_2 = tf.nn.relu(layer_2)  # activation function

    layer_3 = tf.add(tf.matmul(layer_2, hidden_3_layer['weights']), hidden_3_layer['biases'])  # y = x * w + b
    layer_3 = tf.nn.relu(layer_3)  # activation function

    output = tf.matmul(layer_3, output_layer['weights'], output_layer['biases'])

    return output

# Εκπαίδευση του νευρωνικού δικτύου


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10   # Αριθμός επαναλήψεων (forward feed + back propagation)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                x, y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: x, y: y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


train_neural_network(x)



