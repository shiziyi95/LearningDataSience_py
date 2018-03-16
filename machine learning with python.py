
# coding: utf-8

# In[1]:


import numpy as np
x = np.array([[1, 2, 3], [4, 5, 6]])
print("x:\n{}".format(x))


# In[5]:


from scipy import sparse
# Create a 2D NumPy array with a diagonal of ones, and zeros everywhere else
eye = np.eye(4)
print("NumPy array:\n{}".format(eye))


# In[6]:


# Convert the NumPy array to a SciPy sparse matrix in CSR format 
# Only the nonzero entries are stored
sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy sparse CSR matrix:\n{}".format(sparse_matrix))


# In[8]:



data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices))) 
print("COO representation:\n{}".format(eye_coo))


# In[15]:


get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
    # Generate a sequence of numbers from -10 to 10 with 100 steps in between
x = np.linspace(-10, 10, 120)
# Create a second array using sine
y = np.sin(x)
# The plot function makes a line chart of one array against another plt.plot(x, y, marker="x")
plt.plot(x, y, marker="x")


# In[13]:


import pandas as pd
    # create a simple dataset of people
data = {'Name': ["John", "Anna", "Peter", "Linda"],
            'Location' : ["New York", "Paris", "Berlin", "London"],
            'Age' : [24, 13, 53, 33]
}
data_pandas = pd.DataFrame(data)
# IPython.display allows "pretty printing" of dataframes # in the Jupyter notebook
display(data_pandas)


# In[16]:


display(data_pandas[data_pandas.Age > 30])


# In[17]:


import sys
print("Python version: {}".format(sys.version))
import pandas as pd
print("pandas version: {}".format(pd.__version__))
import matplotlib
print("matplotlib version: {}".format(matplotlib.__version__))
import numpy as np
print("NumPy version: {}".format(np.__version__))
import scipy as sp
print("SciPy version: {}".format(sp.__version__))
import IPython
print("IPython version: {}".format(IPython.__version__))
import sklearn
print("scikit-learn version: {}".format(sklearn.__version__))


# In[18]:


from sklearn.datasets import load_iris 
iris_dataset = load_iris()


# In[19]:


print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))


# In[20]:


print(iris_dataset['DESCR'][:193] + "\n...")


# In[21]:


print("Target names: {}".format(iris_dataset['target_names']))


# In[22]:


print("Feature names: \n{}".format(iris_dataset['feature_names']))


# In[23]:


print("Type of data: {}".format(type(iris_dataset['data'])))


# In[24]:


print("Shape of data: {}".format(iris_dataset['data'].shape))


# In[25]:


print("First five columns of data:\n{}".format(iris_dataset['data'][:5]))
#these ---> feature name --->a key(4) --> in irisdataset --->load_iris --->sklearn dataset


# In[26]:


print("Type of target: {}".format(type(iris_dataset['target'])))
#wtf is ndarray and target


# In[27]:


print("Shape of target: {}".format(iris_dataset['target'].shape))
#I suppose know that it is encoded with 0 to 2


# In[28]:


print("Target:\n{}".format(iris_dataset['target']))
#iris['target_names'] array: 0 means setosa, 1 means versicolor, and 2 means virginica.


# In[30]:


# traning data begin
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset['data'], iris_dataset['target'], random_state=0)

