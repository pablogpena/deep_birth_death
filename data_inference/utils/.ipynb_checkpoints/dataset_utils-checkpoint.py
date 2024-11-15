import random
import numpy as np


def shuffle_six_arrays(a, b, c, d, e, f):
    combined = list(zip(a, b, c, d, e, f))
    random.shuffle(combined)
    a_permuted, b_permuted, c_permuted, d_permuted, e_permuted, f_permuted = zip(*combined)
    
    return np.array(a_permuted), np.array(b_permuted), np.array(c_permuted), \
           np.array(d_permuted), np.array(e_permuted), np.array(f_permuted)
