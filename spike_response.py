import numpy as np
import matplotlib.pyplot as plt

T = []
I = []



plt.plot(T, I, label="E12")
plt.ylabel('neuronal activity')
plt.xlabel('time')
plt.legend(loc="upper left")
plt.show()