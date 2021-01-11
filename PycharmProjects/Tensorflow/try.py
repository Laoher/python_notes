import tensorflow as tf
print(tf.__version__)
m1=tf.constant([[4,3]],tf.int32,name = 'first_value')
m2=tf.constant([[2],[3]],tf.int32)
#
product = tf.matmul(m1,m2)

with tf.Session() as sess:  # can also be sess =tf.Session()
    result = sess.run(product)
    print(result)
    sess.run(tf.global_variables_initializer())
    print(sess.run(m2))
    # sess.close() use together with sess = tf.Session(), with...as will close the session automatically

m3 = tf.constant("Guru", tf.string)
m4 = tf.zeros(10)
print(m1)
print(m2)
print(m3)  # Tensor("Const_1:0", shape=(), dtype=string) this 0 should mean 0 dimension I guess
# can output