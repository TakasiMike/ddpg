import tensorflow as tf
tf.enable_eager_execution()

x = tf.get_variable('x', shape=[1, 1], dtype=tf.int32, initializer=tf.constant_initializer([1.]))
y = tf.get_variable('y', shape=[1, 1], dtype=tf.int32, initializer=tf.constant_initializer([1.]))


def partial_f(fun):
        with tf.GradientTape() as tape:

                x_grad, y_grad = tape.gradient(fun, [x, y])
                return x_grad, y_grad


f = 2*tf.square(x) + y
print(partial_f(f))
