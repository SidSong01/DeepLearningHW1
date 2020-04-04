
# coding: utf-8

# # The first function with three models
# ---

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# # The function: y = sin(pi x)/(pi x) + sin(0.5x)
# ---

# In[2]:


X =np.expand_dims(np.arange(0.0, 3.0, 0.01),1)
Y =np.sinc(X) + np.sin(0.5*X)


# #  Define the first neuron network model with 571 parameters
# ### Trainning the first model
# ---

# In[3]:


tf.reset_default_graph()
x = tf.placeholder(tf.float64, [None,1], name='x')
y = tf.placeholder(tf.float64, [None,1], name='y')

hiden_layer1_1 = tf.layers.dense(x, 5, activation= tf.nn.leaky_relu)
hiden_layer2_1 = tf.layers.dense(hiden_layer1_1, 10, activation=tf.nn.leaky_relu)
hiden_layer3_1 = tf.layers.dense(hiden_layer2_1, 10, activation=tf.nn.leaky_relu)
hiden_layer4_1 = tf.layers.dense(hiden_layer3_1, 10, activation=tf.nn.leaky_relu)
hiden_layer5_1 = tf.layers.dense(hiden_layer4_1, 10, activation=tf.nn.leaky_relu)
hiden_layer6_1 = tf.layers.dense(hiden_layer5_1, 10, activation=tf.nn.leaky_relu)
hiden_layer7_1 = tf.layers.dense(hiden_layer6_1, 5, activation=tf.nn.leaky_relu)
output_layer_1 = tf.layers.dense(hiden_layer7_1,1)
Loss_1 =tf.losses.mean_squared_error(y, output_layer_1)
Optimizer_1 = tf.train.AdamOptimizer(learning_rate= 0.05).minimize(Loss_1)
init = tf.global_variables_initializer()

total_parameters_1 = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    #print(variable)
    shape = variable.get_shape()
    print(shape)
    #print(len(shape))
    variable_parameters = 1
    for dim in shape:
        #print(dim)
        variable_parameters *= dim.value
    #print(variable_parameters)
    total_parameters_1 += variable_parameters
print('Total_Parameter number_1 = %s' %total_parameters_1)


print('\n')
with tf.Session() as sess:  
    loss_list_1=[]
    sess.run(init)
    for i in range(0,5000):
        _, loss_val_1 = sess.run([Optimizer_1, Loss_1], feed_dict={x:X, y:Y})
        loss_list_1.append(loss_val_1)
        if i%500 == 0:
            print ('loss = %s' % loss_val_1)
    YP_1 = sess.run(output_layer_1,feed_dict={x:X})


# # Define second model with 572 parameters
# ### Trainning the second model
# ---

# In[7]:


tf.reset_default_graph()
x = tf.placeholder(tf.float64, [None,1], name='x')
y = tf.placeholder(tf.float64, [None,1], name='y')

hiden_layer1_2 = tf.layers.dense(x, 10, activation= tf.nn.leaky_relu)
hiden_layer2_2 = tf.layers.dense(hiden_layer1_2, 18, activation=tf.nn.leaky_relu)
hiden_layer3_2 = tf.layers.dense(hiden_layer2_2, 15, activation=tf.nn.leaky_relu)
hiden_layer4_2 = tf.layers.dense(hiden_layer3_2, 4, activation=tf.nn.leaky_relu)
output_layer_2 = tf.layers.dense(hiden_layer4_2,1, name = 'm2_o')
Loss_2 =tf.losses.mean_squared_error(y, output_layer_2)
Optimizer_2 = tf.train.AdamOptimizer(learning_rate= 0.05).minimize(Loss_2)
init = tf.global_variables_initializer()

total_parameters_2 = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    #print(variable)
    shape = variable.get_shape()
    print(shape)
    #print(len(shape))
    variable_parameters = 1
    for dim in shape:
        #print(dim)
        variable_parameters *= dim.value
    #print(variable_parameters)
    total_parameters_2 += variable_parameters
print('Total_Parameter number = %s' %total_parameters_2)
print('\n')

with tf.Session() as sess:  
    loss_list_2=[]
    sess.run(init)
    for i in range(0,5000):
        _, loss_val_2 = sess.run([Optimizer_2, Loss_2], feed_dict={x:X, y:Y})
        loss_list_2.append(loss_val_2)
        if i%500 == 0:
            print ('loss = %s' % loss_val_2)
    YP_2 = sess.run(output_layer_2,feed_dict={x:X})


# # Define the third model with 571 parameters
# ### Tranning the third model
# ---

# In[5]:


tf.reset_default_graph()
x = tf.placeholder(tf.float64, [None,1], name='x')
y = tf.placeholder(tf.float64, [None,1], name='y')

hiden_layer1_3 = tf.layers.dense(x, 190, activation= tf.nn.leaky_relu)
output_layer_3 = tf.layers.dense(hiden_layer1_3,1)
Loss_3 =tf.losses.mean_squared_error(y, output_layer_3)
Optimizer_3 = tf.train.AdamOptimizer(learning_rate= 0.05).minimize(Loss_3)
init = tf.global_variables_initializer()

total_parameters_3 = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    #print(variable)
    shape = variable.get_shape()
    print(shape)
    #print(len(shape))
    variable_parameters = 1
    for dim in shape:
        #print(dim)
        variable_parameters *= dim.value
    #print(variable_parameters)
    total_parameters_3 += variable_parameters
print('Total_Parameter number = %s' %total_parameters_3)

print('\n')
with tf.Session() as sess:  
    loss_list_3=[]
    sess.run(init)
    for i in range(0,5000):
        _, loss_val_3 = sess.run([Optimizer_3, Loss_3], feed_dict={x:X, y:Y})
        loss_list_3.append(loss_val_3)
        if i%500 == 0:
            print ('loss = %s' % loss_val_3)
    YP_3 = sess.run(output_layer_3,feed_dict={x:X})


# # Plot the outcome of the third models
# ---

# In[8]:


fig,axs = plt.subplots(3,1)
fig.suptitle('Funtion 1 with three models',fontsize = 15)
fig.set_figwidth(15)
fig.set_figheight(15)
axs[0].set_title('First Model')
axs[0].plot(X,Y,X,YP_1)
axs[0].legend(('GtoundTruth','Prediction_1'))
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')

axs[1].set_title('Second Model')
axs[1].plot(X,Y,X,YP_2)
axs[1].legend(('GtoundTruth','Prediction_2'))
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')

axs[2].set_title('Third Model')
axs[2].plot(X,Y,X,YP_3)
axs[2].legend(('GtoundTruth','Prediction_3'))
axs[2].set_xlabel('x')
axs[2].set_ylabel('y')

fig,axs = plt.subplots(3,1)
fig.suptitle('The loss of Function 1 with the three models',fontsize = 15)
fig.set_figwidth(15)
fig.set_figheight(15)
axs[0].set_title('First Model')
axs[0].plot(loss_list_1)
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')

axs[1].set_title('Second Model')
axs[1].plot(loss_list_2)
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')

axs[2].set_title('Third Model')
axs[2].plot(loss_list_3)
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('Loss')


# In[310]:


plt.figure(figsize=(14,14))

plt.plot(X,Y,X,YP_1,X,YP_2,X,YP_3)
plt.title('Three predictions with Ground truth')
plt.legend(('GroundTruth','Prediction1','Prediction2','Prediction3'))
plt.ylabel('y')
plt.xlabel('x')
plt.show()

plt.figure(figsize=(14,14))

epoch = np.arange(5000)
plt.plot(epoch,loss_list_1,epoch,loss_list_2,epoch,loss_list_3)
plt.title('loss of the three models')
plt.legend(('model1','model2','model3'))
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

