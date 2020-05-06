
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.__version__
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
data = input_data.read_data_sets('data/MNIST/', one_hot=True)


# In[3]:


train_num = data.train.num_examples
valid_num = data.validation.num_examples
test_num = data.test.num_examples
img_flatten = 784
img_size = 28
num_classes = 10
print("Size of:")
print("Training Dataset:",train_num)
print("Testing Dataset:",test_num)
print("Validation Dataset:",valid_num)


# In[46]:


tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
y_cls = tf.argmax(y,dimension=1)

conv1 = tf.layers.conv2d(inputs=input_x,filters=8,kernel_size=5,padding="same",activation=tf.nn.relu);
pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=2,strides=2);
conv2 = tf.layers.conv2d(inputs=pool1,filters=8,kernel_size=5,padding="same",activation=tf.nn.relu);
pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=2,strides=2);
flat1 = tf.layers.flatten(pool2);
fc1 = tf.layers.dense(inputs=flat1,units=128,activation=tf.nn.relu);
logits = tf.layers.dense(inputs=fc1,units=num_classes,activation=None);
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits);
loss = tf.reduce_mean(cross_entropy);

softmax = tf.nn.softmax(logits=logits);
pred_op = tf.argmax(softmax,dimension=1);
acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, y_cls), tf.float32));
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001);
train_op = optimizer.minimize(loss);
sens_op = tf.norm(tf.gradients(loss,input_x))

train_loss_list1 = []
train_acc_list1 = []
test_loss_list1 = []
test_acc_list1 = []
sens_list1 = []
init = tf.global_variables_initializer()

Batch_size = [4,8,16,32,64,128,256,512]
with tf.Session() as sess:
    sess.run(init)
    for i in range(len(Batch_size)):
        for j in range(data.train.num_examples // Batch_size[i]):
            x_batch, y_batch = data.train.next_batch(Batch_size[i])
            sess.run(train_op, feed_dict={x: x_batch,y: y_batch})
        train_loss, train_acc = sess.run([loss,acc_op],feed_dict={x:x_batch,y:y_batch})
        train_loss_list1.append(train_loss)
        train_acc_list1.append(train_acc)
        test_loss, test_acc, sens = sess.run([loss,acc_op,sens_op],feed_dict={x:data.test.images,y:data.test.labels})
        test_loss_list1.append(test_loss)
        test_acc_list1.append(test_acc)
        sens_list1.append(sens)
        m = "Batch Size: {0:>4}, Training Loss: {1:>1.4}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.4}, Test Accuracy: {4:>6.1%}, Sensitivity: {5:>1.4}"
        print(m.format(Batch_size[i], train_loss, train_acc, test_loss, test_acc, sens))


# In[48]:


fig,axs=plt.subplots(1,2)
fig.set_figwidth(15)
fig.set_tight_layout('tight')
axs[0].plot(Batch_size,sens_list1,'b')
axs[0].set_xscale('log')
axs[0].set_ylabel('Sensitivity',size=15)
axs[0].set_xlabel('Batch Size(log scale)',size=15)
axs[0].yaxis.label.set_color('b')
axs1 = axs[0].twinx()
axs1.plot(Batch_size, train_loss_list1,'r')
axs1.plot(Batch_size, test_loss_list1,'r--')
axs1.set_xscale('log')
axs1.set_ylabel('Loss',size=15)
axs1.yaxis.label.set_color('r')
axs1.legend(['Train','Test'])
axs1.set_title('Loss, sensitivity vs Batch Size',size=15)

axs[1].plot(Batch_size,sens_list1,'b')
axs[1].set_xscale('log')
axs[1].set_ylabel('Sensitivity',size=15)
axs[1].yaxis.label.set_color('b')
axs2 = axs[1].twinx()
axs2.plot(Batch_size, train_acc_list1,'r')
axs2.plot(Batch_size, test_acc_list1,'r--')
axs2.set_ylabel('Accuracy',size=15)
axs2.yaxis.label.set_color('r')
axs[1].set_xlabel('Batch Size(log scale)',size=15)
axs2.legend(['Train','Test'])
axs2.set_title('Accuracy vs Batch Size',size=15)


# In[14]:


tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
y_cls = tf.argmax(y,dimension=1)

flat1 = tf.layers.flatten(inputs=input_x)
h1 = tf.layers.dense(inputs=flat1,units=128,activation=tf.nn.leaky_relu);
h2 = tf.layers.dense(inputs=h1,units=256,activation=tf.nn.leaky_relu);
h3 = tf.layers.dense(inputs=h1,units=64,activation=tf.nn.leaky_relu);
logits = tf.layers.dense(inputs=h3,units=num_classes,activation=None);
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits);
loss = tf.reduce_mean(cross_entropy);

# Accuracy
softmax = tf.nn.softmax(logits=logits);
pred_op = tf.argmax(softmax,dimension=1);
acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, y_cls), tf.float32));
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001);
train_op = optimizer.minimize(loss);
sens_op = tf.norm(tf.gradients(loss,input_x))

train_loss_list2 = []
train_acc_list2 = []
test_loss_list2 = []
test_acc_list2 = []
sens_list2 = []

init = tf.global_variables_initializer()

Batch_size = [4,8,16,32,64,128,256,512]
with tf.Session() as sess:
    sess.run(init)
    for i in range(len(Batch_size)):
        for j in range(data.train.num_examples // Batch_size[i]):
            x_batch, y_batch = data.train.next_batch(Batch_size[i])
            sess.run(train_op, feed_dict={x: x_batch,y: y_batch})
        train_loss, train_acc = sess.run([loss,acc_op],feed_dict={x:x_batch,y:y_batch})
        train_loss_list2.append(train_loss)
        train_acc_list2.append(train_acc)
        test_loss, test_acc, sens = sess.run([loss,acc_op,sens_op],feed_dict={x:data.test.images,y:data.test.labels})
        test_loss_list2.append(test_loss)
        test_acc_list2.append(test_acc)
        sens_list2.append(sens)
        m = "Batch Size: {0:>4}, Training Loss: {1:>1.4}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.4}, Test Accuracy: {4:>6.1%}, Sensitivity: {5:>1.4}"
        print(m.format(Batch_size[i], train_loss, train_acc, test_loss, test_acc, sens))


# In[36]:


fig,axs=plt.subplots(1,2)
fig.set_figwidth(15)
fig.set_tight_layout('tight')
axs[0].plot(Batch_size,sens_list2,'b')
axs[0].set_xscale('log')
axs[0].set_ylabel('Sensitivity',size=15)
axs[0].set_xlabel('Batch Size(log scale)',size=15)
axs[0].yaxis.label.set_color('b')
axs1 = axs[0].twinx()
axs1.plot(Batch_size, train_loss_list2,'r')
axs1.plot(Batch_size, test_loss_list2,'r--')
axs1.set_xscale('log')
axs1.set_ylabel('Loss',size=15)
axs1.yaxis.label.set_color('r')
axs1.legend(['Train','Test'])
axs1.set_title('Loss, sensitivity vs Batch Size',size=15)

axs[1].plot(Batch_size,sens_list2,'b')
axs[1].set_xscale('log')
axs[1].set_ylabel('Sensitivity',size=15)
axs[1].yaxis.label.set_color('b')
axs2 = axs[1].twinx()
axs2.plot(Batch_size, train_acc_list2,'r')
axs2.plot(Batch_size, test_acc_list2,'r--')
axs2.set_ylabel('Accuracy',size=15)
axs2.yaxis.label.set_color('r')
axs[1].set_xlabel('Batch Size(log scale)',size=15)
axs2.legend(['Train','Test'])
axs2.set_title('Accuracy vs Batch Size',size=15)


# In[17]:


tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
y_cls = tf.argmax(y,dimension=1)

flat1 = tf.layers.flatten(inputs=input_x)
h1 = tf.layers.dense(inputs=flat1,units=128,activation=tf.nn.relu);
h2 = tf.layers.dense(inputs=h1,units=256,activation=tf.nn.relu);
h3 = tf.layers.dense(inputs=h1,units=128,activation=tf.nn.relu);
logits = tf.layers.dense(inputs=h3,units=num_classes,activation=None);
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits);
loss = tf.reduce_mean(cross_entropy);

# Accuracy
softmax = tf.nn.softmax(logits=logits);
pred_op = tf.argmax(softmax,dimension=1);
acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, y_cls), tf.float32));
optimizer = tf.train.AdamOptimizer(learning_rate=0.005);
train_op = optimizer.minimize(loss);
sens_op = tf.norm(tf.gradients(loss,input_x))

train_loss_list3 = []
train_acc_list3 = []
test_loss_list3 = []
test_acc_list3 = []
sens_list3 = []

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(len(Batch_size)):
        for j in range(data.train.num_examples // Batch_size[i]):
            x_batch, y_batch = data.train.next_batch(Batch_size[i])
            sess.run(train_op, feed_dict={x: x_batch,y: y_batch})
        train_loss, train_acc = sess.run([loss,acc_op],feed_dict={x:x_batch,y:y_batch})
        train_loss_list3.append(train_loss)
        train_acc_list3.append(train_acc)
        test_loss, test_acc, sens = sess.run([loss,acc_op,sens_op],feed_dict={x:data.test.images,y:data.test.labels})
        test_loss_list3.append(test_loss)
        test_acc_list3.append(test_acc)
        sens_list3.append(sens)
        m = "Batch Size: {0:>4}, Training Loss: {1:>1.4}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.4}, Test Accuracy: {4:>6.1%}, Sensitivity: {5:>1.4}"
        print(m.format(Batch_size[i], train_loss, train_acc, test_loss, test_acc, sens))


# In[49]:


fig,axs=plt.subplots(1,2)
fig.set_figwidth(15)
fig.set_tight_layout('tight')
axs[0].plot(Batch_size,sens_list3,'b')
axs[0].set_xscale('log')
axs[0].set_ylabel('Sensitivity',size=15)
axs[0].set_xlabel('Batch Size(log scale)',size=15)
axs[0].yaxis.label.set_color('b')
axs1 = axs[0].twinx()
axs1.plot(Batch_size, train_loss_list3,'r')
axs1.plot(Batch_size, test_loss_list3,'r--')
axs1.set_xscale('log')
axs1.set_ylabel('Loss',size=15)
axs1.yaxis.label.set_color('r')
axs1.legend(['Train','Test'])
axs1.set_title('Loss, sensitivity vs Batch Size',size=15)

axs[1].plot(Batch_size,sens_list3,'b')
axs[1].set_xscale('log')
axs[1].set_ylabel('Sensitivity',size=15)
axs[1].yaxis.label.set_color('b')
axs2 = axs[1].twinx()
axs2.plot(Batch_size, train_acc_list3,'r')
axs2.plot(Batch_size, test_acc_list3,'r--')
axs2.set_ylabel('Accuracy',size=15)
axs2.yaxis.label.set_color('r')
axs[1].set_xlabel('Batch Size(log scale)',size=15)
axs2.legend(['Train','Test'])
axs2.set_title('Accuracy, sensitivity vs Batch Size',size=15)


# In[54]:


tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
y_cls = tf.argmax(y,dimension=1)
learning_rate = tf.placeholder(tf.float32)

flat1 = tf.layers.flatten(inputs=input_x)
h1 = tf.layers.dense(inputs=flat1,units=128,activation=tf.nn.relu);
h2 = tf.layers.dense(inputs=h1,units=128,activation=tf.nn.relu);
h3 = tf.layers.dense(inputs=h1,units=128,activation=tf.nn.relu);
h4 = tf.layers.dense(inputs=h1,units=64,activation=tf.nn.relu);
logits = tf.layers.dense(inputs=h4,units=num_classes,activation=None);
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits);
loss = tf.reduce_mean(cross_entropy);

# Accuracy
softmax = tf.nn.softmax(logits=logits);
pred_op = tf.argmax(softmax,dimension=1);
acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, y_cls), tf.float32));
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate);
train_op = optimizer.minimize(loss);
sens_op = tf.norm(tf.gradients(loss,input_x))

train_loss_list4 = []
train_acc_list4 = []
test_loss_list4 = []
test_acc_list4 = []
sens_list4 = []

init = tf.global_variables_initializer()

lr = [0.15,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]
init = tf.global_variables_initializer()
Batchsize = 128

with tf.Session() as sess:
    sess.run(init)
    for i in range(len(lr)):
        for j in range(data.train.num_examples//Batchsize):
            x_batch, y_batch = data.train.next_batch(Batchsize)
            sess.run(train_op, feed_dict={x: x_batch,y: y_batch,learning_rate:lr[i]})
        train_loss, train_acc = sess.run([loss,acc_op],feed_dict={x:x_batch,y:y_batch})
        train_loss_list4.append(train_loss)
        train_acc_list4.append(train_acc)
        test_loss, test_acc, sens = sess.run([loss,acc_op,sens_op],feed_dict={x:data.test.images,y:data.test.labels})
        test_loss_list4.append(test_loss)
        test_acc_list4.append(test_acc)
        sens_list4.append(sens)
        msg = "Learn rate: {0:>4}, Training Loss: {1:>1.4}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.4}, Test Accuracy: {4:>6.1%}, Sensitivity: {5:>1.4}"
        print(msg.format(lr[i], train_loss, train_acc, test_loss, test_acc, sens))


# In[55]:


fig,axs=plt.subplots(1,2)
fig.set_figwidth(15)
fig.set_tight_layout('tight')
axs[0].plot(lr,sens_list4,'b')
axs[0].set_xscale('log')
axs[0].set_ylabel('Sensitivity',size=15)
axs[0].set_xlabel('Learning rate(log scale)',size=15)
axs[0].yaxis.label.set_color('b')
axs1 = axs[0].twinx()
axs1.plot(lr, train_loss_list4,'r')
axs1.plot(lr, test_loss_list4,'r--')
axs1.set_xscale('log')
axs1.set_ylabel('Loss',size=15)
axs1.yaxis.label.set_color('r')
axs1.legend(['Train','Test'])
axs1.set_title('Loss, sensitivity vs Learning rate',size=15)

axs[1].plot(lr,sens_list4,'b')
axs[1].set_xscale('log')
axs[1].set_ylabel('Sensitivity',size=15)
axs[1].yaxis.label.set_color('b')
axs2 = axs[1].twinx()
axs2.plot(lr, train_acc_list4,'r')
axs2.plot(lr, test_acc_list4,'r--')
axs2.set_ylabel('Accuracy',size=15)
axs2.yaxis.label.set_color('r')
axs[1].set_xlabel('Learning rate(log scale)',size=15)
axs2.legend(['Train','Test'])
axs2.set_title('Accuracy,sensitivity vs Learning rate',size=15)


# In[57]:


tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
y_cls = tf.argmax(y,dimension=1)
learning_rate = tf.placeholder(tf.float32)

conv1 = tf.layers.conv2d(inputs=input_x,filters=8,kernel_size=5,padding="same",activation=tf.nn.relu);
pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=2,strides=2);
flat1 = tf.layers.flatten(pool1);
fc1 = tf.layers.dense(inputs=flat1,units=128,activation=tf.nn.relu);
logits = tf.layers.dense(inputs=fc1,units=num_classes,activation=None);
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits);
loss = tf.reduce_mean(cross_entropy);

softmax = tf.nn.softmax(logits=logits);
pred_op = tf.argmax(softmax,dimension=1);
acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, y_cls), tf.float32));
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001);
train_op = optimizer.minimize(loss);
sens_op = tf.norm(tf.gradients(loss,input_x))

train_loss_list5 = []
train_acc_list5 = []
test_loss_list5 = []
test_acc_list5 = []
sens_list5 = []
init = tf.global_variables_initializer()

lr = [0.15,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]
init = tf.global_variables_initializer()
Batchsize = 128

with tf.Session() as sess:
    sess.run(init)
    for i in range(len(lr)):
        for j in range(data.train.num_examples//Batchsize):
            x_batch, y_batch = data.train.next_batch(Batchsize)
            sess.run(train_op, feed_dict={x: x_batch,y: y_batch,learning_rate:lr[i]})
        train_loss, train_acc = sess.run([loss,acc_op],feed_dict={x:x_batch,y:y_batch})
        train_loss_list5.append(train_loss)
        train_acc_list5.append(train_acc)
        test_loss, test_acc, sens = sess.run([loss,acc_op,sens_op],feed_dict={x:data.test.images,y:data.test.labels})
        test_loss_list5.append(test_loss)
        test_acc_list5.append(test_acc)
        sens_list5.append(sens)
        msg = "Learn rate: {0:>4}, Training Loss: {1:>1.4}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.4}, Test Accuracy: {4:>6.1%}, Sensitivity: {5:>1.4}"
        print(msg.format(lr[i], train_loss, train_acc, test_loss, test_acc, sens))


# In[58]:


fig,axs=plt.subplots(1,2)
fig.set_figwidth(15)
fig.set_tight_layout('tight')
axs[0].plot(lr,sens_list4,'b')
axs[0].set_xscale('log')
axs[0].set_ylabel('Sensitivity',size=15)
axs[0].set_xlabel('Learning rate(log scale)',size=15)
axs[0].yaxis.label.set_color('b')
axs1 = axs[0].twinx()
axs1.plot(lr, train_loss_list4,'r')
axs1.plot(lr, test_loss_list4,'r--')
axs1.set_xscale('log')
axs1.set_ylabel('Loss',size=15)
axs1.yaxis.label.set_color('r')
axs1.legend(['Train','Test'])
axs1.set_title('Loss, sensitivity vs Learning rate',size=15)

axs[1].plot(lr,sens_list4,'b')
axs[1].set_xscale('log')
axs[1].set_ylabel('Sensitivity',size=15)
axs[1].yaxis.label.set_color('b')
axs2 = axs[1].twinx()
axs2.plot(lr, train_acc_list4,'r')
axs2.plot(lr, test_acc_list4,'r--')
axs2.set_ylabel('Accuracy',size=15)
axs2.yaxis.label.set_color('r')
axs[1].set_xlabel('Learning rate(log scale)',size=15)
axs2.legend(['Train','Test'])
axs2.set_title('Accuracy,sensitivity vs Learning rate',size=15)

