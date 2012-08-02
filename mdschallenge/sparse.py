'''
Created on Jul 31, 2012

@author: Paolo
'''

from scipy.sparse.construct import eye

v = eye(100000, 100000, 0, 'd')

v.T