import tensorflow as tf
tf.enable_eager_execution()


class PartDer(object):

        def __init__(self, x, y, fun):
                self.x = x
                self.y = y
                self.fun = fun

        def part_der(self):

            with tf.GradientTape as tape:
                    x_grad, y_grad = tape.gradient(self.fun, [self.x, self.y])
                    return x_grad, y_grad


with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

x = PartDer(tf.get_variable('x', shape=[1, 1], initializer=tf.constant_initializer([1.])))
y = PartDer(tf.get_variable('y', shape=[1, 1], initializer=tf.constant_initializer([1.])))

f = 2*tf.square(x) + y
print(PartDer.part_der(f))

