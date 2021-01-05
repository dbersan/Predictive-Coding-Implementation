'''
Generates fake data into CSV files
'''

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import matplotlib.pyplot as plt

# Select function to sample
FUNCTION    = f1
NOISE_LEVEL = 0.8
VISUALIZE   = True
RANGE_MIN   = -3
RANGE_MAX   = 3
STEP        = 0.4

# Functions to be sampled

def f1(x,y):
    # f = x^2 + y^2
    return x**2 + y**2

def create_csv(filename, data):
    separator = ','
    with open(filename+'.csv', 'w') as f:
        keys = list(data)
        data_points = len(data[keys[0]])

        for k in range(len(keys)):
            f.write(f'"{keys[k]}"')
            if k < len(keys)-1:
                f.write(separator)
        
        f.write('\n')

        for i in range(data_points):
            for k in range(len(keys)):
                f.write(f'{data[keys[k]][i]}')
                if k < len(keys)-1:
                    f.write(separator)
                else:
                    f.write(f'\n')

    f.close()

def generate_data(generator_function, range_min, range_max, step, noise_level=0.2,visualize = False):
    # Generates data samples for dim_in = 2 and dim_out = 1 functions
    assert range_min < range_max
    assert step <= abs(range_max- range_min)

    x = y = np.arange(range_min, range_max, step)
    X, Y = np.meshgrid(x, y)
    xs = np.ravel(X)
    ys = np.ravel(Y)
    zs = np.array(f1(xs, ys))

    if noise_level>0.0:
        noise = np.random.normal(0,noise_level,len(zs))
        zs+=noise

    Z = zs.reshape(X.shape)

    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    return {'x': xs, 'y': ys, 'f': zs}

filename = 'generated_'+FUNCTION.__name__
data = generate_data(FUNCTION, RANGE_MIN, RANGE_MAX, STEP, noise_level=NOISE_LEVEL, visualize=VISUALIZE)
create_csv(filename, data)

