
# coding: utf-8

# In[1]:


# imports
import numpy as np
import scipy as scp
import scipy.integrate as sint


# In[2]:


# define beta
beta = np.ones((3,3))
gamma = 1


# In[3]:


def generate_S(n = 3, scale = 1):
    arr = np.ones((n))
    ex = np.array([float(10)**(j*scale) for j in range(1,n+1)])
    return arr*ex


# In[4]:


def generate_I(n = 3, initial = 1):
    arr = np.array([initial*float(j) for j in range(n)])
    return arr


# In[5]:


def generate_R(n = 3):
    arr = np.zeros((n))
    return arr.astype(float)


# In[6]:


def initial_counts(n = 3, scale = 1, initial = 1):
    S = generate_S(n,scale)
    I = generate_I(n, initial)
    R = generate_R(n)
    
    S = S - I
    
    return np.array([S,I,R])


# In[7]:


test = initial_counts(n = 3, scale = 1, initial = 1)


# In[8]:


def f(t, y):
    B = np.array([[1,1,1], [1,1,1], [1,1,1]]).astype(float)
    gamma = 1
    n = 3
    S = np.array([y[:n].astype(float)])
    I = np.transpose([y[n:2*n].astype(float)])
    R = y[2*n:].astype(float)
    
    oneS = np.matmul(np.ones((n,1)), S)
    
    sdot = -1*np.matmul(B,np.matmul(oneS,I)) #3x1 in this case
    idot = sdot - gamma*I #3x1 in this case
    rdot = gamma*I #3x1 in this case
    
    bad = [entry.flatten() for entry in [sdot, idot, rdot]]
  
    return np.array(bad).flatten()


# In[9]:


y_0 = initial_counts()
y = sint.solve_ivp(f, [0,10], np.array(range(9)))

