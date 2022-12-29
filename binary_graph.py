import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt

with open('/Users/jiangshanyu/Desktop/predicted_results.pkl', 'rb') as f:
    data = pickle.load(f)

trial_list = []
mua_list = []
for index in data.index:
    if index[0] not in trial_list:
        trial_list.append(index[0])
    if index[1] not in mua_list:
        mua_list.append(index[1])

trial_list.sort()
mua_list.sort()

# Towards
for trial in trial_list:
    towardsDict = {}
    for mua in range(1, 101):
        if mua not in mua_list:
            towardsDict[mua] = 0
        else:
            w = data.loc[[(trial, mua, 'T')]]
            # m = w[w['True Values'] == w['Predicted Values']]
            if (w['True Values'] == w['Predicted Values']).bool():
                towardsDict[mua] = 1
            else:
                towardsDict[mua] = 0
    binaryMap = np.array(list(towardsDict.values())).reshape((10, 10))
    # save binary map
    filename = 'binaryGraphs/towards/trial' + str(trial)
    with open(filename, 'wb') as f:
        pickle.dump(binaryMap, f)
    # break

# Away
for trial in range(1, 710):
    AwayDict = {}
    for mua in range(1, 101):
        if mua not in mua_list:
            AwayDict[mua] = 0
        else:
            w = data.loc[[(trial, mua, 'A')]]
            # m = w[w['True Values'] == w['Predicted Values']]
            if (w['True Values'] == w['Predicted Values']).bool():
                AwayDict[mua] = 1
            else:
                AwayDict[mua] = 0
    binaryMap = np.array(list(AwayDict.values())).reshape((10, 10))
    # save binary map
    filename = 'binaryGraphs/away/trial' + str(trial)
    with open(filename, 'wb') as f:
        pickle.dump(binaryMap, f)
    # break

# # plot graph
# fig, ax = plt.subplots(figsize=(10,10))
# im = ax.imshow(data, cmap='Greys_r')
# plt.show()




