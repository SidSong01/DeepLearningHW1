
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
tf.logging.set_verbosity(tf.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)


# In[17]:


#Function used to plot 9 images in a 3x3 grid, and writing the true and predicted classes below each image.
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# In[18]:


# Get the first images from the test-set.
images = data.test.images[0:9]

# Get the true classes for those images.
cls_true = data.test.cls[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)


# In[4]:


train_batch_size = 64
n_batch = data.train.num_examples // train_batch_size

learning_rate = 1e-4


# In[5]:


data.test.cls = np.argmax(data.test.labels, axis=1)
# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10


# In[15]:


tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# layer_conv1
net = tf.layers.conv2d(inputs=x_image, name='layer_conv1', padding='same',
                       filters=16, kernel_size=5, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

# layer_conv2
net = tf.layers.conv2d(inputs=net, name='layer_conv2', padding='same',
                       filters=36, kernel_size=5, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

print(net)

net = tf.layers.flatten(net)

print(net)

net = tf.layers.dense(inputs=net, name='layer_fc1',
                      units=128, activation=tf.nn.relu)
logits = tf.layers.dense(inputs=net, name='layer_fc_out',
                      units=num_classes, activation=None)
print(logits)

y_pred = tf.nn.softmax(logits=logits)
y_pred_cls = tf.argmax(y_pred, dimension=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)

loss = tf.reduce_mean(cross_entropy)

opt = tf.train.AdamOptimizer(learning_rate)
optimizer = opt.minimize(loss)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#trainable_var_list = []
#for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
#    trainable_var_list.append(var)
#    print(var)

#trainable_var_list = tf.trainable_variables()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_list_1 = []
    acc_list_1 = []
    for i in range(50):
        for batch in range(n_batch):
            # Get a batch of training examples.
            # x_batch now holds a batch of images and
            # y_true_batch are the true labels for those images.
            x_batch, y_true_batch = data.train.next_batch(train_batch_size)

            # Put the batch into a dict with the proper names
            # for placeholder variables in the TensorFlow graph.
            feed_dict_train = {x: x_batch,
                               y_true: y_true_batch}

            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.
            sess.run(optimizer, feed_dict=feed_dict_train)
        los, acc = sess.run([loss, accuracy], feed_dict={x:data.test.images, y_true:data.test.labels})
        loss_list_1.append(los)
        acc_list_1.append(acc)

        print("Epoch: " + str(i) + ", acc: " + str(acc) + ", loss:" + str(los))


# In[9]:


tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# layer_conv1
net = tf.layers.conv2d(inputs=x_image, name='layer_conv1', padding='same',
                       filters=16, kernel_size=5, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

# layer_conv2
net = tf.layers.conv2d(inputs=net, name='layer_conv2', padding='same',
                       filters=64, kernel_size=5, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

print(net)

net = tf.layers.flatten(net)

print(net)

net = tf.layers.dense(inputs=net, name='layer_fc1',
                      units=16, activation=tf.nn.relu)
logits = tf.layers.dense(inputs=net, name='layer_fc_out',
                      units=num_classes, activation=None)
print(logits)

y_pred = tf.nn.softmax(logits=logits)
y_pred_cls = tf.argmax(y_pred, dimension=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)

loss = tf.reduce_mean(cross_entropy)

opt = tf.train.AdamOptimizer(learning_rate)
optimizer = opt.minimize(loss)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#trainable_var_list = []
#for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
#    trainable_var_list.append(var)
#    print(var)

#trainable_var_list = tf.trainable_variables()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_list_2 = []
    acc_list_2 = []
    for i in range(50):
        for batch in range(n_batch):
            # Get a batch of training examples.
            # x_batch now holds a batch of images and
            # y_true_batch are the true labels for those images.
            x_batch, y_true_batch = data.train.next_batch(train_batch_size)

            # Put the batch into a dict with the proper names
            # for placeholder variables in the TensorFlow graph.
            feed_dict_train = {x: x_batch,
                               y_true: y_true_batch}

            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.
            sess.run(optimizer, feed_dict=feed_dict_train)
        los, acc = sess.run([loss, accuracy], feed_dict={x:data.test.images, y_true:data.test.labels})
        loss_list_2.append(los)
        acc_list_2.append(acc)

        print("Epoch: " + str(i) + ", acc: " + str(acc) + ", loss:" + str(los))


# In[11]:


tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# layer_conv1
net = tf.layers.conv2d(inputs=x_image, name='layer_conv1', padding='same',
                       filters=16, kernel_size=5, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

# layer_conv2
net = tf.layers.conv2d(inputs=net, name='layer_conv2', padding='same',
                       filters=16, kernel_size=5, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

# layer_conv3
net = tf.layers.conv2d(inputs=net, name='layer_conv3', padding='same',
                       filters=16, kernel_size=5, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

print(net)

net = tf.layers.flatten(net)

print(net)

net = tf.layers.dense(inputs=net, name='layer_fc1',
                      units=64, activation=tf.nn.relu)
logits = tf.layers.dense(inputs=net, name='layer_fc_out',
                      units=num_classes, activation=None)
print(logits)

y_pred = tf.nn.softmax(logits=logits)
y_pred_cls = tf.argmax(y_pred, dimension=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)

loss = tf.reduce_mean(cross_entropy)

opt = tf.train.AdamOptimizer(learning_rate)
optimizer = opt.minimize(loss)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#trainable_var_list = []
#for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
#    trainable_var_list.append(var)
#    print(var)

#trainable_var_list = tf.trainable_variables()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_list_3 = []
    acc_list_3 = []
    for i in range(50):
        for batch in range(n_batch):
            # Get a batch of training examples.
            # x_batch now holds a batch of images and
            # y_true_batch are the true labels for those images.
            x_batch, y_true_batch = data.train.next_batch(train_batch_size)

            # Put the batch into a dict with the proper names
            # for placeholder variables in the TensorFlow graph.
            feed_dict_train = {x: x_batch,
                               y_true: y_true_batch}

            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.
            sess.run(optimizer, feed_dict=feed_dict_train)
        los, acc = sess.run([loss, accuracy], feed_dict={x:data.test.images, y_true:data.test.labels})
        loss_list_3.append(los)
        acc_list_3.append(acc)

        print("Epoch: " + str(i) + ", acc: " + str(acc) + ", loss:" + str(los))


# In[16]:


# Plot training  loss values
import matplotlib.pyplot as plt
plt.plot(loss_list_1)
plt.plot(loss_list_2)
plt.plot(loss_list_3)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(('Model1','Model2','Model3'))
plt.show()

plt.plot(acc_list_1)
plt.plot(acc_list_2)
plt.plot(acc_list_3)
plt.title('Model accuracy')
plt.ylabel('acc')
plt.xlabel('Epoch')
plt.legend(('Model1','Model2','Model3'))
plt.show()

