import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt

with open('/Users/jiangshanyu/Desktop/LFP_TW/vistraw/results/classification_NoScale/tsCNN1_20221129/predictions-fold-0.pkl', 'rb') as f:
    data = pickle.load(f)


input_path_away = '/Users/jiangshanyu/Desktop/layout.mat'
mat = scipy.io.loadmat(input_path_away)
x = mat['plxarray'].flatten()

my_list = []
for i in data.index:
    if i[1] not in my_list:
        my_list.append(i[1])

my_dict = {}
for i in my_list:
    w = data.xs(i, level='mua_id')
    t = w[w['True Values'] == w['Predicted Values']]
    acc = t.shape[0]/w.shape[0]
    my_dict[x[i]] = acc

heat_map = {}
for j in range(100):
    if j not in my_dict.keys():
        heat_map[j] = 0
    else:
        heat_map[j] = my_dict[j]

heat_map = np.array(list(heat_map.values())).reshape((10,10))

fig, ax = plt.subplots(figsize=(10,10))
im = ax.imshow(heat_map)
plt.title( "2-D Heat Map" )
ax.figure.colorbar(im, ax = ax, shrink=0.5 )
plt.show()
