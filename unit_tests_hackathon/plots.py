import numpy as np
import matplotlib.pyplot as printlog

path = '/scratch/snx3000/mibianco/results_c2ray/test_eurohack23/'
with open(path+"IonRates_2.000.pkl","rb") as f:
    data = pkl.load(f)

plt.imshow(data[50])
plt.savefig(path+'test.png', bbox_inches='tight')
plt.clf()