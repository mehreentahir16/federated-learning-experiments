import pickle
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt


with open('50-epoch/non-IID/90-stragglers/fedavg_42_iid[0]_E[50].pkl', 'rb') as handle:
    a = pickle.load(handle)
l1 = a[1]


with open('50-epoch/non-IID/90-stragglers/fedprox_42_iid[0]_E[50].pkl', 'rb') as handle:
    b = pickle.load(handle)
l2 = b[1]
print("l2",l2)


with open('50-epoch/non-IID/90-stragglers/fedPD_42_iid[0]_E[50].pkl', 'rb') as handle:
    c = pickle.load(handle)
l3 = c[1]


with open('50-epoch/non-IID/90-stragglers/scaffold_42.pkl', 'rb') as handle:
    d = pickle.load(handle)
l4 = d[1]


with open('50-epoch/non-IID/90-stragglers/fedMed_42_iid[0]_E[50].pkl', 'rb') as handle:
    e = pickle.load(handle)
l5 = e[1]

with open('50-epoch/non-IID/90-stragglers/qfedavg_42_iid[0]_E[50].pkl', 'rb') as handle:
    f = pickle.load(handle)
l6 = f[1]

df = pd.DataFrame({"FedAvg": l1,
                 "FedProx" : l2,
                 "FedPD" : l3,
                 "SCAFFOLD" : l4,
                 "FedMed" : l5,
                 "qFedAvg" : l6})
# print(len(l1))
# x = []
# for i in range(len(l1)):
#   x.append(i)
fig = sns.lineplot(data=df)

# fig.set_xlabel("Number of Rounds")
# fig.set_ylabel("Test Accuracy")
# fig, ax = plt.subplots()
# #ax.plot(x_axis, y_axis, 'tab:'+plt_color)
fig.set(xlabel='Number of Rounds', ylabel='Average Test Accuracy')
# ax.plot(x, l1, label = "FedAvg")
# ax.plot(x, l2, label = "FedProx")
# ax.plot(x, l3, label = "FedPD")
# ax.plot(x, l4, label = "SCAFFOLD")
# ax.plot(x, l5, label = "FedMed")
# ax.plot(x, l6, label = "qFedAvg")

plt.ylim(0,1)

# plt.legend()
plt.savefig('50-epoch/non-IID/90-stragglers/accuracy-non-iid.png', format='png')
plt.show()