


import cv2
import numpy as np
import os
from random import shuffle
import matplotlib.pyplot as plt 

training_directory = '/home/helpologist/junction_detection/junc_frames_return/'
testing_directory = '/home/helpologist/Downloads/test-images/'
img_size = 80
learn_rate = 1e-3
epochs = 1
model_name = 'junction_detection_none_included-{}-{}.model'.format(learn_rate, '2-conv-basic')

font = cv2.FONT_HERSHEY_SIMPLEX
# In[93]:


#[x, x] = [0,1] saum and [1,0] nav
#saum.45.jpg
def label_image(img):
	word_label = img.split('.')[-3]
	if word_label == 'junc1': return [1,0]
	elif word_label == 'none': return [0,1]
	#elif word_label == 'none': return [0,0,1]


# In[94]:


def create_train_data():
	train_data = []
	image_list = [x for x in os.listdir(training_directory) if x.endswith('.jpg')]
	for img in image_list:
		label = label_image(img)
		path = os.path.join(training_directory, img)
		img = cv2.resize(cv2.imread(path, 0), (img_size, img_size))
		train_data.append([np.array(img), np.array(label)])
		
	shuffle(train_data)
	np.save('train_data.npy', train_data)
	return train_data
	


# In[95]:


def process_test_data():
	test_data = []
	image_list = [x for x in os.listdir(testing_directory) if x.endswith('.jpg')]
	for img in image_list:
		path = os.path.join(testing_directory, img)
		img = cv2.resize(cv2.imread(path, 0), (img_size, img_size))
		test_data.append(np.array(img))
		
	np.save('test_Data.npy', test_data)
	return test_data, image_list


# In[96]:


train_data = create_train_data()

# In[97]:


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.estimator import regression

convnet = input_data(shape=[None, img_size, img_size, 1], name='input')





convnet = conv_2d(convnet, 32, 5, activation='softmax')
convnet = max_pool_2d(convnet,2)

convnet = conv_2d(convnet, 64, 5, activation='softmax')
convnet = max_pool_2d(convnet,2)

convnet = fully_connected(convnet, 1024, activation='softmax')
convnet = dropout(convnet,0.8)
convnet = fully_connected(convnet, 2, activation='softmax')




# convnet = conv_2d(convnet, 24, 5,5, activation='softmax')
# convnet = conv_2d(convnet, 36, 5,5, activation='softmax')
# convnet = conv_2d(convnet, 48, 5,5, activation='softmax')
# convnet = conv_2d(convnet, 64, 5,5, activation='softmax')
# convnet = conv_2d(convnet, 64, 5,5, activation='softmax')
# convnet = dropout(convnet,0.8)
# #convnet = flatten(convnet)
# convnet = fully_connected(convnet, 100, activation='softmax')
# convnet = fully_connected(convnet, 50, activation='softmax')
# convnet = fully_connected(convnet, 10, activation='softmax')

# convnet = fully_connected(convnet, 2)
convnet = regression(convnet,optimizer='adam',learning_rate=learn_rate, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)


# In[98]:


train = train_data[:-50] #all but last 50
test = train_data[-50:]


# In[99]:


X = np.array([i[0] for i in train]).reshape(-1,img_size,img_size,1) #pixel data in train...1st content, and reshaped to fit tflearn
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,img_size,img_size,1) #pixel data in train...1st content, and reshaped to fit tflearn
test_y = [i[1] for i in test]


# In[ ]:


model.fit({'input' : X}, {'targets' : Y}, n_epoch = epochs, validation_set=({'input':test_x},{'targets':test_y}),batch_size=70,snapshot_step=500, show_metric=True, run_id=model_name)


# In[ ]:

model.save(model_name + '.tflearn')
training_data = np.load('train_data.npy')




test_data, image_list = process_test_data()
j=0
for data in test_data:
	orig = data
	data = data.reshape(img_size, img_size, 1)
	model_out = model.predict([data])[0]
	if np.argmax(model_out) == 0:
		print ("junc detected for {}".format(image_list[j]))
		path = os.path.join(testing_directory, image_list[j])
		image = cv2.imread(path)
		height, width, channels = image.shape
		cv2.putText(image,'JUNC',(10,50), font, 2,(0,0,255),2)
		cv2.imwrite("junc%d.jpg"%j, image)
		j += 1
	else:
		print ("none")
		path = os.path.join(testing_directory, image_list[j])
		im = cv2.imread(path)
		height, width, channels = im.shape
		cv2.putText(im,'NONE',(0,50), font, 2,(0,255,0),2)
		cv2.imwrite("none%d.jpg"%j, im)
		j += 1






# fig = plt.figure()

# for num, data in enumerate(training_data[:12]):
# 	img_num = data[1]
# 	img_data = data[0]

# 	y = fig.add_subplot(3,4,num+1)
# 	orig = img_data
# 	data = img_data.reshape(img_size, img_size, 1)

# 	model_out = model.predict([data])[0]
# 	if np.argmax(model_out) ==  0: 
# 		str_label = 'junc1'
# 	elif np.argmax(model_out) == 1: 
# 		str_label = 'none'
# 	# else:
# 	# 	str_label = 'none'

# 	y.imshow(orig, cmap='gray')
# 	plt.title(str_label)
# 	y.axes.get_xaxis().set_visible(False)
# 	y.axes.get_yaxis().set_visible(False)

# plt.show()