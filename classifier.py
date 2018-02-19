'''Softmax-Classifier for CIFAR-10'''


import tensorflow as tf
import os
from keras.datasets import cifar10
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
baseDir = os.path.dirname(os.path.abspath('__file__')) + '/'
classesName = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
xVal = xTrain[49000:, :].astype(np.float)
yVal = np.squeeze(yTrain[49000:, :])
xTrain = xTrain[:49000, :].astype(np.float)
yTrain = np.squeeze(yTrain[:49000, :])
yTest = np.squeeze(yTest)
xTest = xTest.astype(np.float)



# Parameter definitions
batch_size = 100
learning_rate = 0.0000006
max_steps = 1400

print ('Train image shape:    {0}'.format(xTrain.shape))
print ('Train label shape:    {0}'.format(yTrain.shape))
print ('Validate image shape: {0}'.format(xVal.shape))
print ('Validate label shape: {0}'.format(yVal.shape))
print ('Test image shape:     {0}'.format(xTest.shape))
print ('Test label shape:     {0}'.format(yTest.shape))

# Show some CIFAR10 images
plt.subplot(221)
plt.imshow(xTrain[0])
plt.axis('off')
plt.title(classesName[yTrain[0]])
plt.subplot(222)
plt.imshow(xTrain[1])
plt.axis('off')
plt.title(classesName[yTrain[1]])
plt.subplot(223)
plt.imshow(xVal[0])
plt.axis('off')
plt.title(classesName[yVal[1]])
plt.subplot(224)
plt.imshow(xTest[0])
plt.axis('off')
plt.title(classesName[yTest[0]])
plt.savefig(baseDir+'svm0.png')
plt.clf()

meanImage = np.mean(xTrain, axis=0)
xTrain -= meanImage
xVal -= meanImage
xTest -= meanImage

# Reshape data from channel to rows
xTrain = np.reshape(xTrain, (xTrain.shape[0], -1))
xVal = np.reshape(xVal, (xVal.shape[0], -1))
xTest = np.reshape(xTest, (xTest.shape[0], -1))

# Add bias dimension columns


print ('Train image shape after add bias column:   {0}'.format(xTrain.shape))
print ('Val image shape after add bias column:     {0}'.format(xVal.shape))
print ('Test image shape after add bias column:    {0}'.format(xTest.shape))


# xTrain=np.reshape(xTrain, (xTrain.shape[0], -1))
# xTest = np.reshape(xTest, (xTest.shape[0], -1))
# yTrain = np.squeeze(yTrain)
# yTest = np.squeeze(yTest)


# Uncommenting this line removes randomness
# You'll get exactly the same result on each run
# np.random.seed(1)

# Prepare data

# -----------------------------------------------------------------------------
# Prepare the TensorFlow graph
# (We're only defining the graph here, no actual calculations taking place)
# -----------------------------------------------------------------------------

# Define input placeholders
images_placeholder = tf.placeholder(tf.float32, shape=[None, 3072])
labels_placeholder = tf.placeholder(tf.int64, shape=[None])

# Define variables (these are the values we want to optimize)
weights = tf.Variable(tf.zeros([3072, 10]))
biases = tf.Variable(tf.zeros([10]))
FLAGS= tf.app.flags.FLAGS


# Define the classifier's result
logits = tf.matmul(images_placeholder, weights) + biases
labels_placeholder= tf.reshape (labels_placeholder, [-1,1])

regularization_loss = 0.5*tf.reduce_sum(tf.square(weights))
hinge_loss = tf.reduce_mean( tf.losses.hinge_loss(labels_placeholder,logits=logits[:,1:]))
loss = regularization_loss + hinge_loss
#hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([batch_size,1]),
#1 - labels_placeholder*logits));
#loss = regularization_loss + hinge_loss;
# Define the loss function
#loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
 # labels=labels_placeholder))

# Define the training operation
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

sess=tf.InteractiveSession()
tf.global_variables_initializer().run()


# Operation comparing prediction with true label
correct_prediction = tf.equal(tf.argmax(logits, 1), labels_placeholder)

# Operation calculating the accuracy of our predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# -----------------------------------------------------------------------------
# Run the TensorFlow graph
# -----------------------------------------------------------------------------
lossHistory = []

# Repeat max_steps times
for i in range(max_steps):

    s=np.arange(xTrain.shape[0])
    np.random.shuffle(s)
    xTr=xTrain[s]
    yTr=yTrain[s]
    batch_xs=xTr[:100]
    batch_ys=yTr[:100]
    main_loss,_= sess.run([loss,train_step], feed_dict={images_placeholder:batch_xs,labels_placeholder:batch_ys})
    train_accuracy = sess.run( accuracy, feed_dict={images_placeholder: xTrain, labels_placeholder: yTrain})
    test_accuracy = sess.run( accuracy, feed_dict={images_placeholder: xTest, labels_placeholder: yTest})

    lossHistory.append(main_loss)
    if i % 100 == 0 and len(lossHistory) is not 0:
      print( 'Loop {0} loss {1} '.format(i, lossHistory[i]) )
      print ('Train_Accuracy {0} Test_Accuracy {1}'.format(train_accuracy,test_accuracy))


    #print(lossHistory[i])



