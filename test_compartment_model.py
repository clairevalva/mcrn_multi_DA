
# coding: utf-8

# In[244]:


# imports
import numpy as np
import scipy as scp
import scipy.integrate as sint
import matplotlib.pyplot as plt


# In[2]:


def generate_S(n = 3, scale = 1):
    arr = np.ones((n))
    ex = np.array([float(10)**(j*scale) for j in range(1,n+1)])
    return arr*ex


# In[3]:


def generate_I(n = 3, initial = 1):
    arr = np.array([initial*float(j) for j in range(n)])
    return arr


# In[4]:


def generate_R(n = 3):
    arr = np.zeros((n))
    return arr.astype(float)


# In[217]:


def initial_counts(n = 3, scale = 1, initial = 1):
    S = generate_S(n,scale)
    I = generate_I(n, initial)
    R = generate_R(n)
    
    S = S - I
    
    return np.array([S,I,R])


# In[218]:


def compute_sdot(S,I,R,beta,C):
    '''
    S,I ,R, are n by 1 vectors, beta is a constant,
    C is an n by n matrix
    '''
    
    inv_pop = np.divide(np.ones_like(S + I + R, dtype = float), S + I + R)
    
    sdot = [-beta*S[i]*np.sum(np.multiply(C[i],
                                          np.multiply(I,inv_pop)))
            for i in range(len(S))]
    return np.array(sdot)    


# In[219]:


def compute_idot(S,I,R,beta,C,gamma):
    sdot = -1*compute_sdot(S,I,R,beta,C) 
    return sdot - gamma*I


# In[220]:


def compute_rdot(I,gamma):
    return gamma*I


# In[367]:


beta = 0.5
n = 2
C = [[10.,.8],[.8,1.]]


# In[353]:


S = [33500,168000-33499]
I = [0,1]
R = [0,0]
y_0 = np.array([S,I,R]).flatten()


# In[354]:


def f(t,y):
    S = y[:n]
    I = y[n:2*n]
    R = y[2*n:3*n]
    
    sdot = compute_sdot(S,I,R,beta,C)
    idot = compute_idot(S,I,R,beta,C,gamma)
    rdot = compute_rdot(I,gamma)
    
    return np.array([sdot,idot,rdot]).flatten()


# In[368]:


pleasework = sint.solve_ivp(f,[0,20],y_0, max_step = 0.01)


# In[369]:


shaped = np.reshape(pleasework.y, (3,n,len(pleasework.t)))


# In[370]:


#comp1 = shaped[:,0,:]


# In[374]:


plt.clf()
plt.scatter(pleasework.t,np.add(shaped[1,0,:], shaped[1,1,:]),label = "I, city")
plt.scatter(pleasework.t,shaped[1,0,:],label = "I, university")
plt.legend()
plt.show()


# In[362]:


#sums = np.sum(comp1,axis = 0)


# In[363]:


compartment = 1
plt.clf()
plt.scatter(pleasework.t,shaped[0,compartment,:],label = "S")
plt.scatter(pleasework.t,shaped[1,compartment,:],label = "I")
plt.scatter(pleasework.t,shaped[2,compartment,:],label = "R")
plt.legend()
plt.show()


# In[364]:


compartment = 0
plt.clf()
plt.scatter(pleasework.t,shaped[0,compartment,:],label = "S")
plt.scatter(pleasework.t,shaped[1,compartment,:],label = "I")
plt.scatter(pleasework.t,shaped[2,compartment,:],label = "R")
plt.legend()
plt.show()

