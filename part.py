import tensorflow as tf

tf.enable_eager_execution()


x = tf.get_variable('w', shape=[1, 1], initializer=tf.constant_initializer([1.]))
y = tf.get_variable('y', shape=[1, 1], initializer=tf.constant_initializer([1.]))

with tf.GradientTape() as tape:
    fun = 2*tf.square(x) + y
    x_grad, y_grad = tape.gradient(fun, [x, y])
    print(fun.numpy(), x_grad.numpy(), y_grad.numpy())



