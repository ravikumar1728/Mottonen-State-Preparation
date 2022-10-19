#!/usr/bin/env python
# coding: utf-8

# # Problem Statement :
# 
# ## Task 3 Mottonen state preparation
# Implement the Mottonen state preparation of any dataset you have for at most one 8-element vector.
# 
# 
# def state_prep(Optional[list,array]: input_vector):
# \<br>
#      “””
#      input_vector: List, array that contain float values of size 2^n
#      Return the  mottomen state preparation of the input_vector
#      “””
# 
#      # use a framework that works with quantum circuits, qiskit, cirq, pennylane, etc. 
#      # define a quantum circuit to convert the vector in a quantum circuit
#      # define the Mottonen state
# 
# 
# # consider print your quantum circuit
# 
# 
# Bonus: 
# Consider a state vector of size 5,7,10 how you can implement a vector of size different to 2^n.
# 
# References: 
# 
# [1] Transformation of quantum states using uniformly controlled rotations 
# https://arxiv.org/pdf/quant-ph/0407010.pdf 
# 

# # I have discussed the implementation and its overiew in a separate pdf document "Implementation and Overview" in the 

# In[1]:


import numpy as np    # requires two libraries namely Numpy and qiskit
from qiskit import QuantumCircuit,QuantumRegister,execute
from qiskit.providers.aer import QasmSimulator


# In[2]:


def alpha_y(ar,k,j) :    # gives the \alpha 's required for the amplitude encoding
  ar = np.abs(ar)
  m = int(2**(k-1))
  a = 0

  for l in range(m) :
    a+= ar[(2*(j+1)-1)*m+l]**2

  mk = int(2**k)
  b = 0

  for l in range(mk) :
    b+= ar[j*mk+l]**2

  if b != 0 :
    r = np.sqrt(a/b)
  else :
    r = 0.0

  return 2*np.arcsin(r) 

def g(i) :  # binary grey code return functions
  return i ^ (i >> 1) 

def M(k) : # the transformation matrix which takes \alpha 's to \theta 's 
  
  n = 2**k
  M = np.zeros([n,n])

  for i in range(n) :
    for j in range(n) :

      M[i,j] = 2**(-k)*(-1)**(bin(j&g(i)).count('1'))

  return M

def theta(M,alphas) : # \alpha_y --- > \theta_y , \theta = M @ \alpha_y 
    
  return M@alphas


def ind(k) : # gives the control index required while appling the required c-x gates
  
  n = 2**k
  code = [g(i) for i in range(n)]
  
  control = []
  
  for i in range(n-1) :
    control.append(int(np.log2(code[i]^code[i+1])))
  control.append(int(np.log2(code[n-1]^code[0])))
  return control

# k-controlled uniform rotation gate decomposed into the c-nots and rotational y's, R_y(\theta)

def k_controlled_uniform_rotation_y(qc,alpha_k,control_qubits,target_qubit) : 
  
  k = len(control_qubits)
  n = 2**k
  thetas = theta(M(k),alpha_k)

  if k == 0:
    qc.ry(thetas[0],target_qubit)
  else :
    control_index = ind(k)
    for i in range(n) :
      qc.ry(thetas[i],target_qubit)
      qc.cx(control_qubits[k-1-control_index[i]],target_qubit)


# In[3]:



def state_prep(input_vector) :
    
    n = int(np.log2(len(input_vector)))
    state = ( 1/np.linalg.norm(input_vector) )* input_vector

    q = QuantumRegister(n)  # the quantum register initialized with all |0>'s with the circuit
    qc = QuantumCircuit(q)
    
    for k in range(n):
        alpha_k = [ alpha_y(state,n-k,j) for j in range(2**(k))]
        k_controlled_uniform_rotation_y(qc,alpha_k,q[:k],q[k])
        qc.barrier()
        
    print(qc.draw())    
    print("\n \n")
    print("Let's verify the results using QuasmSimulator")
    
    qc.save_statevector()
    backend = QasmSimulator()
    backend_options = {'method': 'statevector'}
    job = execute(qc, backend, backend_options=backend_options)
    job_result = job.result()
    
    print("\n QuasmSimulators result : \n",sorted(np.abs(job_result.get_statevector(qc))))
    print("\n Input normalized vector : \n",sorted(state))


# ### Let's test the circuit using a 8 element vector.
#  I choose a random vector of length 8.

# In[4]:


N = 8
input_vector =  np.random.random([N])
input_vector


# In[5]:


state_prep(input_vector)


#  ### what if length is not of 2^n form.
#  
#  I have tried trivial solution by adding zeros to this type of vector so it attains 2^n form.
#  I know there is redundacy in this way, which can be removed and made more streamlined.

# In[6]:


N = 5
input_vector2 =  np.random.random([N])


# In[7]:


def state_prep_2(input_vector) :
    
    n = int(np.ceil(np.log2(len(input_vector))))
    input_vector = np.array(list(input_vector) + [0.0]*(2**n-len(input_vector)))
    state = ( 1/np.linalg.norm(input_vector) )* input_vector

    q = QuantumRegister(n)  # the quantum register initialized with all |0>'s with the circuit
    qc = QuantumCircuit(q)
    
    for k in range(n):
        alpha_k = [ alpha_y(state,n-k,j) for j in range(2**(k))]
        k_controlled_uniform_rotation_y(qc,alpha_k,q[:k],q[k])
        qc.barrier()
        
    print(qc.draw())    
    print("\n \n")
    print("Let's verify the results using QuasmSimulator")
    
    qc.save_statevector()
    backend = QasmSimulator()
    backend_options = {'method': 'statevector'}
    job = execute(qc, backend, backend_options=backend_options)
    job_result = job.result()
    
    print("\n QuasmSimulators result : \n",sorted(np.round(np.abs(job_result.get_statevector(qc)),10)))
    print("\n Input normalized vector : \n",sorted(state))


# In[8]:


state_prep_2(input_vector2)


# # Acknowledgement :
# 
# I am grateful the following resources which I found very helpful during the implementation.
# 
# 
# [1] Transformation of quantum states using uniformly controlled rotations 
# https://arxiv.org/pdf/quant-ph/0407010.pdf 
#     
# [2] Efficient decomposition of quantum gates
# https://arxiv.org/pdf/quant-ph/0312218.pdf
#     
# [3] Pennylane (pennylane.templates.state_preparations.mottonen)
# 
# [4] Uniform rotation gate implemntation by dc-qiskit-algorithms 
# https://dc-qiskit-algorithms.readthedocs.io/
# 
# Last but not the least, QOSF for giving me this opportunity to implement such good problems.    
