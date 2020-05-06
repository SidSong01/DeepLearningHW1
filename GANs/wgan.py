
import numpy as np
import tensorflow as tf
import pickle
from dataSet import ReadFromTFRecord, DataBatch, ImgShow
import matplotlib.pyplot as plt
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize
from keras.datasets import cifar10
from numpy import asarray
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import exp

def calculate_inception_score(p_yx, eps=1E-16):
    # calculate p(y)
    p_y = expand_dims(p_yx.mean(axis=0), 0)
    # kl divergence for each image
    kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = mean(sum_kl_d)
    # undo the logs
    is_score = exp(avg_kl_d)
    return is_score

 
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)
 
# calculate frechet inception distance
def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    fid = fid/10
    return fid

def batch_preprocess(data_batch):

    batch = sess.run(data_batch)
    
    batch_images = np.reshape(batch, [-1, 3, 32, 32]).transpose((0, 2, 3, 1))
    batch_images = batch_images * 2 - 1
    return  batch_images


def Dir():
    import os
    if not os.path.isdir('ckpt'):
        os.mkdir('ckpt')
    if not os.path.isdir('trainLog'):
        os.mkdir('trainLog')

real_shape = [-1,32,32,3]
data_total = 5000 
batch_size = 64 
noise_size = 128 
max_iters = 35000 
learning_rate = 5e-5
CRITIC_NUM = 5
CLIP = [-0.1,0.1]

def GeNet(z, channel, is_train=True):

    with tf.variable_scope("generator", reuse=(not is_train)):

        layer1 = tf.layers.dense(z, 4 * 4 * 512)
        layer1 = tf.reshape(layer1, [-1, 4, 4, 512])
        layer1 = tf.layers.batch_normalization(layer1, training=is_train,)
        layer1 = tf.nn.relu(layer1)

        layer2 = tf.layers.conv2d_transpose(layer1, 256, 3, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(0, 0.02),
                                            bias_initializer=tf.random_normal_initializer(0, 0.02))
        layer2 = tf.layers.batch_normalization(layer2, training=is_train)
        layer2 = tf.nn.relu(layer2)

        layer3 = tf.layers.conv2d_transpose(layer2, 128, 3, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(0, 0.02),
                                            bias_initializer=tf.random_normal_initializer(0, 0.02))
        layer3 = tf.layers.batch_normalization(layer3, training=is_train)
        layer3 = tf.nn.relu(layer3)

        layer4 = tf.layers.conv2d_transpose(layer3, 64, 3, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(0, 0.02),
                                            bias_initializer=tf.random_normal_initializer(0, 0.02))
        layer4 = tf.layers.batch_normalization(layer4, training=is_train)
        layer4 = tf.nn.relu(layer4)

        logits = tf.layers.conv2d_transpose(layer4, channel, 3, strides=1, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(0, 0.02),
                                            bias_initializer=tf.random_normal_initializer(0, 0.02))
        # outputs
        outputs = tf.tanh(logits)

        return logits,outputs


def DiNet(inputs_img, reuse=False, GAN = False,GP= False,alpha=0.2):

    with tf.variable_scope("discriminator", reuse=reuse):

        layer1 = tf.layers.conv2d(inputs_img, 128, 3, strides=2, padding='same')
        if GP is False:
            layer1 = tf.layers.batch_normalization(layer1, training=True)
        layer1 = tf.nn.leaky_relu(layer1,alpha=alpha)

        layer2 = tf.layers.conv2d(layer1, 256, 3, strides=2, padding='same')
        if GP is False:
            layer2 = tf.layers.batch_normalization(layer2, training=True)
        layer2 = tf.nn.leaky_relu(layer2, alpha=alpha)

        layer3 = tf.layers.conv2d(layer2, 512, 3, strides=2, padding='same')
        if GP is False:
            layer3 = tf.layers.batch_normalization(layer3, training=True)
        layer3 = tf.nn.leaky_relu(layer3, alpha=alpha)
        layer3 = tf.reshape(layer3, [-1, 4*4* 512])

        logits = tf.layers.dense(layer3, 1)

        if GAN:
            outputs = None
        else:
            outputs = tf.sigmoid(logits)

        return logits, outputs

#inputs
inputs_real = tf.placeholder(tf.float32, [None, real_shape[1], real_shape[2], real_shape[3]], name='inputs_real')
inputs_noise = tf.placeholder(tf.float32, [None, noise_size], name='inputs_noise')
_,g_outputs = GeNet(inputs_noise, real_shape[3], is_train=True)
_,g_test = GeNet(inputs_noise, real_shape[3], is_train=False)

d_logits_real, _ = DiNet(inputs_real,GAN=True)
d_logits_fake, _ = DiNet(g_outputs,GAN=True,reuse=True)

#-original loss function
g_loss = tf.reduce_mean(-d_logits_fake)

d_loss = tf.reduce_mean(d_logits_fake - d_logits_real)


train_vars = tf.trainable_variables()
g_vars = [var for var in train_vars if var.name.startswith("generator")]
d_vars = [var for var in train_vars if var.name.startswith("discriminator")]

# Optimizer
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    g_train_opt = tf.train.RMSPropOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)
    d_train_opt = tf.train.RMSPropOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)

# clip
d_clip_opt = [tf.assign(var, tf.clip_by_value(var, CLIP[0], CLIP[1])) for var in d_vars]

# read TFR
[data,label] = ReadFromTFRecord(sameName= r'./TFR/class1-*',isShuffle= False,datatype= tf.float64,
                                labeltype= tf.int64,isMultithreading= True)

[data_batch,label_batch] = DataBatch(data,label,dataSize= 32*32*3,labelSize= 1,
                                                   isShuffle= True,batchSize= 64)
# save
GenLog = []
losses = []
saver = tf.train.Saver(var_list=[var for var in tf.trainable_variables()
                                 if var.name.startswith("generator")],max_to_keep=5)

with tf.Session() as sess:

    Dir()

    init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)


    for steps in range(max_iters):
        steps += 1

        if steps < 25 or steps % 500 == 0:
            critic_num = 100
        else:
            critic_num = CRITIC_NUM

        for i in range(CRITIC_NUM):
            batch_images = batch_preprocess(data_batch)  # images
            batch_noise = np.random.normal(size=(batch_size, noise_size))  # noise(normal)
            _ = sess.run(d_train_opt, feed_dict={inputs_real: batch_images,
                                                 inputs_noise: batch_noise})
            sess.run(d_clip_opt)

        batch_images = batch_preprocess(data_batch)  # images
        batch_noise = np.random.normal(size=(batch_size, noise_size))  # noise(normal)
        _ = sess.run(g_train_opt, feed_dict={inputs_real: batch_images,
                                             inputs_noise: batch_noise})

        if steps % 5 == 1:

            train_loss_d = d_loss.eval({inputs_real: batch_images,
                                        inputs_noise: batch_noise})
            train_loss_g = g_loss.eval({inputs_real: batch_images,
                                        inputs_noise: batch_noise})
            losses.append([train_loss_d, train_loss_g,steps])


            batch_noise = np.random.normal(size=(batch_size, noise_size))
            gen_samples = sess.run(g_test, feed_dict={inputs_noise: batch_noise})
            genLog = (gen_samples[0:11] + 1) / 2
            GenLog.append(genLog)


            print('step {}...'.format(steps),
                  "Discriminator Loss: {:.4f}...".format(train_loss_d),
                  "Generator Loss: {:.4f}...".format(train_loss_g))

        if steps % 300 ==0:
            saver.save(sess, './ckpt/generator.ckpt', global_step=steps)

    coord.request_stop()
    coord.join(threads)
    
with open('./trainLog/loss_variation.loss', 'wb') as l:
    losses = np.array(losses)
    pickle.dump(losses,l)
    print('loss saved')

with open('./trainLog/GenLog.log', 'wb') as g:
    pickle.dump(GenLog, g)
    print('GenLog saved..')

#output
with open('./trainLog/GenLog.log', 'rb') as f:

    GenLog = pickle.load(f)
    GenLog = np.array(GenLog)
    ImgShow(GenLog,[-1],10)

with open(r'./trainLog/loss_variation.loss','rb') as l:
    losses = pickle.load(l)
    fig, ax = plt.subplots(figsize=(20, 7))
    plt.plot(losses.T[2],losses.T[0], label='Discriminator  Loss')
    plt.plot(losses.T[2],losses.T[1], label='Generator Loss')
    plt.title("Training Losses")
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

with tf.Session() as sess:

    meta_graph = tf.train.import_meta_graph('./ckpt/generator.ckpt-34800.meta')
    meta_graph.restore(sess,tf.train.latest_checkpoint('./ckpt'))
    graph = tf.get_default_graph()
    inputs_noise = graph.get_tensor_by_name("inputs_noise:0")
    d_outputs_fake = graph.get_tensor_by_name("generator/Tanh:0")
    sample_noise= np.random.normal(size=(10, 128))
    gen_samples = sess.run(d_outputs_fake,feed_dict={inputs_noise: sample_noise})
    gen_samples = [(gen_samples[0:11]+1)/2]
    ImgShow(gen_samples, [0], 10)
    for i in range(10):
        img=gen_samples[0][i]
        plt.imshow(img)
        plt.axis('off')
        plt.savefig("wgan_img/car_%d.png" % i)
        plt.show

# conditional probabilities for high quality images
p_yx = gen_samples[0]
score = calculate_inception_score(p_yx)
print('inception score of WGAN: %.3f' %score)

# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

images2=gen_samples[0]

# load cifar10 images
(images1, _), (_, _) = cifar10.load_data()
shuffle(images1)
images1 = images1[:10]
images2=images2[:10]
print('Loaded', images1.shape, images2.shape)
# convert integer to floating point values
images1 = images1.astype('float32')
images2 = images2.astype('float32')
# resize images
images1 = scale_images(images1, (299,299,3))
images2 = scale_images(images2, (299,299,3))
print('Scaled', images1.shape, images2.shape)
# pre-process images
images1 = preprocess_input(images1)
images2 = preprocess_input(images2)
# calculate fid
fid = calculate_fid(model, images1, images2)
print('FID of WGAN: %.3f' % fid)