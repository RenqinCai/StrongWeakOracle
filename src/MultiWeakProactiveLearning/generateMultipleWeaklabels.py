import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score as ACC


input1 = np.genfromtxt('../../dataset/sensorType/rice_hour_sdh', delimiter=',')
pn1 = [i.strip().split('\\')[-1][:-5] for i in open('../../dataset/sensorType/rice_pt_sdh').readlines()]

input2 = np.genfromtxt('../../dataset/sensorType/keti_hour_sum', delimiter=',')
input3 = np.genfromtxt('../../dataset/sensorType/sdh_hour_rice', delimiter=',')
input2 = np.vstack((input2,input3))
pn2 = [i.strip().split('+')[-1][:-5] for i in open('../../dataset/sensorType/sdh_pt_rice').readlines()]

input3 = np.genfromtxt('../../dataset/sensorType/soda_hour_sum', delimiter=',')
pn3 = [i.strip().split('+')[-1][:-5] for i in open('../../dataset/sensorType/soda_pt').readlines()]

X_fd = [input1, input2, input3]
X_fn = [pn1, pn2, pn3]

Y = [np.unique(X[:,-1]) for X in X_fd]
common = set(Y[0]) & set(Y[1])
common &= set(Y[2])
#common = np.array(common)

X_fd = [input1, input2]
X_fn = [pn1, pn2]

fd = []
fn = []
for d,n in zip(X_fd,X_fn):
    #fd_tmp.append( np.array(list(filter(lambda x: x[-1] in common, X_fd))) )
    fd_tmp = []
    fn_tmp = []
    for fd_,fn_ in zip(d,n):
        if fd_[-1] in common:
            fd_tmp.append(fd_)
            fn_tmp.append(fn_)

    fd.append(np.array(fd_tmp))
    fn.append(fn_tmp)

X_fd = fd
X_fn = fn

rf = RFC(n_estimators=100, criterion='entropy')
bldg = ['rice','sdh','soda']
# for i in range(len(X_fd)):
i = 0
source = [X_fd[j] for j in range(len(X_fd)) if j!=i]
train = np.vstack(source)
train_fd = train[:,:-1]
train_label = train[:, -1]
test_fd, test_label = X_fd[i][:,:-1], X_fd[i][:,-1]
#print (train_fd.shape, train_label.shape, test_fd.shape, test_label.shape)

rf.fit(train_fd, train_label)
preds = rf.predict(test_fd)

print (ACC(preds, test_label))
assert(len(test_label) == len(X_fn[i]))

sourceName = bldg[1]
targetName = bldg[0]

dataDir = "../../dataset/sensorType/sdh_soda_rice"
transferLabelFileName = "transferLabel_"+sourceName+"--"+targetName+".txt"
transferLabelFileName = os.path.join(dataDir, transferLabelFileName)
f = open(transferLabelFileName, "w")

totalInstanceNum = len(test_label)
f.write("auditorLabel"+"\t"+"transferLabel"+"\t"+"trueLabel\n")
for instanceIndex in range(totalInstanceNum):
    transferLabel = preds[instanceIndex]
    trueLabel = test_label[instanceIndex]

    # print(transferLabel, trueLabel)

    if transferLabel == trueLabel:
        f.write("1.0"+"\t")
    else:
        f.write("0.0"+"\t")

    f.write(str(transferLabel)+"\t"+str(trueLabel))
    f.write("\n")

f.close()


# X_fd = [input1, input3]
# X_fn = [pn1, pn3]

# fd = []
# fn = []
# for d,n in zip(X_fd,X_fn):
#     #fd_tmp.append( np.array(list(filter(lambda x: x[-1] in common, X_fd))) )
#     fd_tmp = []
#     fn_tmp = []
#     for fd_,fn_ in zip(d,n):
#         if fd_[-1] in common:
#             fd_tmp.append(fd_)
#             fn_tmp.append(fn_)

#     fd.append(np.array(fd_tmp))
#     fn.append(fn_tmp)

# X_fd = fd
# X_fn = fn

# rf = RFC(n_estimators=100, criterion='entropy')
# bldg = ['rice','sdh','soda']
# # for i in range(len(X_fd)):
# i = 0
# source = [X_fd[j] for j in range(len(X_fd)) if j!=i]
# train = np.vstack(source)
# train_fd = train[:,:-1]
# train_label = train[:, -1]
# test_fd, test_label = X_fd[i][:,:-1], X_fd[i][:,-1]
# #print (train_fd.shape, train_label.shape, test_fd.shape, test_label.shape)

# rf.fit(train_fd, train_label)
# preds = rf.predict(test_fd)

# print (ACC(preds, test_label))
# assert(len(test_label) == len(X_fn[i]))

# sourceName = bldg[2]
# targetName = bldg[0]

# dataDir = "../../dataset/sensorType/sdh_soda_rice"
# transferLabelFileName = "transferLabel_"+sourceName+"--"+targetName+".txt"
# transferLabelFileName = os.path.join(dataDir, transferLabelFileName)
# f = open(transferLabelFileName, "w")

# totalInstanceNum = len(test_label)
# f.write("auditorLabel"+"\t"+"transferLabel"+"\t"+"trueLabel\n")
# for instanceIndex in range(totalInstanceNum):
#     transferLabel = preds[instanceIndex]
#     trueLabel = test_label[instanceIndex]

#     # print(transferLabel, trueLabel)

#     if transferLabel == trueLabel:
#         f.write("1.0"+"\t")
#     else:
#         f.write("0.0"+"\t")

#     f.write(str(transferLabel)+"\t"+str(trueLabel))
#     f.write("\n")

# f.close()

# df = pd.DataFrame( np.vstack( (np.array(preds == test_label).astype(int), preds, test_label) ).T )
# df.to_csv('%s_%s_labels.csv'%(bldg[i],bldg[2]) )

# with open('%s_names'%bldg[i], 'w') as outfile:
#         outfile.write('\n'.join(X_fn[i]) + '\n')

    #ptn = [i.strip().split('\\')[-1][:-5] for i in open('./soda_pt_sdh').readlines()]
    #test_fn = get_name_features(ptn)