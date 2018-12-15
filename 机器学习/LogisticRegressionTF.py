import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from datetime import datetime
import os

m = 1000
X_moons, y_moons = make_moons(m, noise=0.1, random_state=42)

# plt.plot(X_moons[y_moons == 1, 0], X_moons[y_moons == 1, 1], 'go', label="Positive")
# plt.plot(X_moons[y_moons == 0, 0], X_moons[y_moons == 0, 1], 'r^', label="Negative")
# plt.legend()
# plt.show()

X_moons_with_bias = np.c_[np.ones((m,1)), X_moons]
y_moons_column_vector = y_moons.reshape(-1, 1)

test_ratio = 0.2
test_size = int(m * test_ratio)
X_train = X_moons_with_bias[:-test_size]
X_test = X_moons_with_bias[-test_size:]
y_train = y_moons_column_vector[:-test_size]
y_test = y_moons_column_vector[-test_size:]

def random_batch(X_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return X_batch, y_batch

X_train_enhanced = np.c_[X_train,
						np.square(X_train[:, 1]),
						np.square(X_train[:, 2]),
						X_train[:, 1] ** 3,
						X_train[:, 2] ** 3
						]
X_test_enhanced = np.c_[X_test,
                        np.square(X_test[:, 1]),
                        np.square(X_test[:, 2]),
                        X_test[:, 1] ** 3,
                        X_test[:, 2] ** 3]

def logistic_regression(X, y, initializer=None, seed=42, learning_rate=0.01):
	# Create Graph
	n_inputs_including_bias = int(X.get_shape()[1])
	with tf.name_scope("logistic_regression"):
		with tf.name_scope("model"):
			if initializer is None:
				initializer = tf.random_uniform([n_inputs_including_bias, 1], -1.0, 1.0, seed=seed)
			theta = tf.Variable(initializer, name="theta")
			logits = tf.matmul(X, theta, name="logits")
			y_proba = tf.sigmoid(logits)
		with tf.name_scope("train"):
			loss = tf.losses.log_loss(y, y_proba, scope="loss")
			optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
			training_op = optimizer.minimize(loss)
			loss_summary = tf.summary.scalar('log_loss', loss) # binary log
		with tf.name_scope("init"):
			init = tf.global_variables_initializer()
		with tf.name_scope("save"):
			saver = tf.train.Saver()
	return y_proba, loss, training_op, loss_summary, init, saver



def log_dir(prefix=""):
	now = datetime.now().strftime("%Y%m%d%H%M%S")  # datetime.now() is local time 
												   # dateime.utcnow()
	root_logdir = "tf_logs"
	if prefix:
		prefix += "-"
	name = prefix + "run-" + now
	return "{}/{}/{}".format(os.getcwd(),root_logdir, name)

n_inputs = 2 + 4
logdir = log_dir("logreg")
# print(logdir)
X = tf.placeholder(tf.float32, shape=(None, n_inputs + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

y_proba, loss, training_op, loss_summary, init, saver = logistic_regression(X, y)

file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph()) # binary log

n_epochs = 10001
batch_size = 50
n_batches = int(np.ceil(m / batch_size))

checkpoint_path = os.getcwd()+"/tmp/my_logreg_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = os.getcwd()+"/my_logreg_model"


with tf.Session() as sess:
	if os.path.isfile(checkpoint_epoch_path):
		with open(checkpoint_epoch_path, "rb") as f:
			start_epoch = int(f.read())
		print("Traning was interrtupted. Continuing at epoch", start_epoch)
		saver.restore(sess, checkpoint_path)
	else:
		start_epoch = 0
		sess.run(init)

	for epoch in range(start_epoch, n_epochs):
		for batch_index in range(n_batches):
			X_batch, y_batch = random_batch(X_train_enhanced, y_train, batch_size)
			sess.run(training_op, feed_dict={X:X_batch, y:y_batch})

		loss_val, summary_str = sess.run([loss, loss_summary], feed_dict={X:X_test_enhanced, y: y_test})
		file_writer.add_summary(summary_str, epoch)
		if epoch % 500 == 0:
			print("Epoch:", epoch, "\tLoss:", loss_val)
			saver.save(sess, checkpoint_path)
			with open(checkpoint_epoch_path, "wb") as f:
				f.write(b"%d" % (epoch + 1))

	saver.save(sess, final_model_path)
	y_proba_val = y_proba.eval(feed_dict = {X: X_test_enhanced, y: y_test})
	os.remove(checkpoint_epoch_path)


# reset_graph()

# n_inputs = 2
# X = tf.placeholder(tf.float32, shape=(None, n_inputs + 1), name="X")
# y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
# theta = tf.Variable(tf.random_uniform([n_inputs + 1, 1], -1.0, 1.0, seed=42), name="theta")
# logits = tf.matmul(X, theta, name="logits")
# y_proba = 1 / (1 + tf.exp(-logits)) # sigmoid

# epsilon = 1e-7  # to avoid an overflow when computing the log
# loss = -tf.reduce_mean(y * tf.log(y_proba + epsilon) + (1 - y) * tf.log(1 - y_proba + epsilon))

# loss = tf.losses.log_loss(y, y_proba) 

# learning_rate = 0.01
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
# training_op = optimizer.minimize(loss)

# init = tf.global_variables_initializer()

# n_epochs = 1000
# batch_size = 50
# n_batches = int(np.ceil(m / batch_size))

# with tf.Session() as sess:
# 	sess.run(init)

# 	for epoch in range(n_epochs):
# 		for batch_index in range(n_batches):
# 			# X_batch, y_batch = random_batch(X_train, y_train, batch_size)
# 			# sess.run(traning_op, feed_dict={X: X_batch, y: y_batch})
# 			X_batch, y_batch = random_batch(X_train, y_train, batch_size)
# 			sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
# 		# loss_val = loss.eval({X:X_test, y:y_test})
# 		loss_val = loss.eval(feed_dict={X: X_test, y: y_test})
# 		if epoch % 100 == 0:
# 			print("Epoch:", epoch, "\tLoss:", loss_val)

# 	y_proba_val = y_proba.eval(feed_dict={X: X_test, y: y_test})

# reset_graph()