
# coding: utf-8

# In[2]:


from sympy import symbols
import sympy as sp


# In[3]:


aa = symbols('i_x'), symbols('i_y'), symbols('i_z')
bb = symbols('j_x'), symbols('j_y'), symbols('j_z')
cc = symbols('k_x'), symbols('k_y'), symbols('k_z')


# In[22]:


def vec_diff(a, b):
    return a[0]-b[0], a[1]-b[1], a[2]-b[2]

def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def norm_vec(a, b):
    return sp.sqrt(dot(a, b))

def norm_self(a):
    return norm_vec(a, a)

def dist(a, b):
    return norm_self(vec_diff(a, b))

def cross(a, b):
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]
    return x, y, z
    
def cos_ijk(i, j, k):
    a = vec_diff(j, i)
    b = vec_diff(k, i)
    top = dot(a, b) 
    bot = norm_self(a) * norm_self(b)
    return top / bot

def sin_ijk(i, j, k):
    a = vec_diff(j, i)
    b = vec_diff(k, i)
    top = norm_self(cross(a, b))
    bot = norm_self(a) * norm_self(b)
    return top / bot

def f_C(a, b):
    r_ij = dist(a, b)
    return 0.5 * sp.cos((sp.pi*r_ij)/Rc) + 0.5

def radial(a, b):
    return sp.exp(-eta*((dist(a,b)-Rs) ** 2))*f_C(a,b)

def angular(a, b, c):
    first = (1 + cos_ijk(a,b,c)*sp.cos(Ts) + sin_ijk(a,b,c)*sp.sin(Ts)) ** zeta
    second = sp.exp(-eta*((dist(a,b) + dist(a,c))/2 - Rs) ** 2)
    third = f_C(a,b) * f_C(a,c)
    return 2**(1-zeta)*first*second*third


# In[21]:


Rc = symbols('R_Rc')
Rs = symbols('R_Rs[r_idx]')
eta = symbols('R_eta')
print(sp.ccode(radial(aa, bb)))
print("---")
A0 = sp.ccode(sp.diff(radial(aa, bb), bb[0]))
A1 = sp.ccode(sp.diff(radial(aa, bb), bb[1]))
A2 = sp.ccode(sp.diff(radial(aa, bb), bb[2]))
print(A0)
print("--")
print(A1)
print("--")
print(A2)
print(A0==A1)


# In[25]:




print('\n--ANGULAR--\n')
Rc = symbols('A_Rc')
Rs = symbols('A_Rs[s]')
Ts = symbols('A_thetas[t]')
zeta = symbols('A_zeta')
eta = symbols('A_eta')
print(sp.ccode(angular(aa, bb, cc)))
print('-----')
print(sp.ccode(sp.diff(angular(aa, bb, cc), cc[0])))
print('-----')
print(sp.ccode(sp.diff(angular(aa, bb, cc), cc[1])))
print('-----')
print(sp.ccode(sp.diff(angular(aa, bb, cc), cc[2])))


# In[158]:


# a,b,c = vec_diff(x0, y0, z0, x1, y1, z1)
# sp.diff(norm(a,b,c), x0)
# print(cos_ijk(9x0,y0,z0, x1,y1,z1, x2,y2,z2))

# sp.diff(cos_ijk(aa, bb, cc), aa[0])
# sp.diff(sin_ijk(aa, bb, cc), aa[0])
# sp.diff(f_C(aa, bb), aa[0])
# sp.ccode(sp.diff(angular(aa, bb, cc), aa[0]))
sp.ccode(sp.diff(angular(aa, bb, cc), cc[0]))


# In[66]:


sp.pi


# In[101]:





# In[102]:


sp.exp


# In[112]:


def foo(a, b):
    return a**2, b**2

sp.diff(foo(aa[0], bb[0])[0], aa[0])
sp.diff(foo(aa[0], bb[0])[1], bb[0])

