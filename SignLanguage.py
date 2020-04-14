#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[33]:


train = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test.csv')


# In[34]:


train.head()


# In[35]:


train.shape


# In[36]:


from IPython.display import Image
Image("american_sign_language.PNG")


# In[38]:


labels = train['label'].values


# In[39]:


unique_val = np.array(labels)
np.unique(unique_val)


# In[40]:


plt.figure(figsize = (18,8))
sns.countplot(x =labels)


# In[41]:


train.drop('label', axis = 1, inplace = True)


# In[42]:


images = train.values
images = np.array([np.reshape(i, (28, 28)) for i in images])
images = np.array([i.flatten() for i in images])


# In[43]:


from sklearn.preprocessing import LabelBinarizer
label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(labels)


# In[44]:


labels


# In[45]:


plt.imshow(images[0].reshape(28,28))


# In[46]:


from sklearn.model_selection import train_test_split


# In[47]:


x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.3, random_state = 101)


# In[52]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout


# In[53]:


batch_size = 128
num_classes = 24
epochs = 50


# In[54]:


x_train = x_train / 255
x_test = x_test / 255


# In[55]:


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)


# In[56]:


x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


# In[57]:


plt.imshow(x_train[0].reshape(28,28))


# In[58]:


model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28 ,1) ))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.20))

model.add(Dense(num_classes, activation = 'softmax'))


# In[59]:


model.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


# In[60]:


history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=batch_size)


# In[66]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','test'])

plt.show()


# In[67]:


test_labels = test['label']


# In[68]:


test.drop('label', axis = 1, inplace = True)


# In[91]:


test_images = test.values
test_images = np.array([np.reshape(i, (28, 28)) for i in test_images])
test_images = np.array([i.flatten() for i in test_images])


# In[70]:


test_labels = label_binrizer.fit_transform(test_labels)


# In[71]:


test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)


# In[72]:


test_images.shape


# In[73]:


y_pred = model.predict(test_images)


# In[74]:


from sklearn.metrics import accuracy_score


# In[75]:


accuracy_score(test_labels, y_pred.round())


# In[ ]:




