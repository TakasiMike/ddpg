import tensorflow as tf

x = tf.Variable(1., tf.float32, name="x")
y = tf.Variable(1., tf.float32, name="y")


g = tf.exp(x*x) + tf.log(y) + 5

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
init.run()
result = g.eval()
print(result)
