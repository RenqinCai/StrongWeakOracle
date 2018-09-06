import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
import os
import pandas as pd
from matplotlib.ticker import ScalarFormatter,AutoMinorLocator
import matplotlib as mpl

plt.style.use("./AL.mplstyle")
plt.close('all')
fig, ax = plt.subplots()

def readAccFile(accFile):

    f=open(accFile)

    meanAcc = [0.0, 0.0, 0.0]
    varAcc = [0.0, 0.0, 0.0] 
    acc = []

    foldNum = 10
    roundNum = 10
    for rawLine in f:
        line = rawLine.strip().split("\t")
        lineLen = len(line)

        roundNum = lineLen

        accFoldList = []
        for eleIndex in range(lineLen):
            accFoldList.append(float(line[eleIndex]))
#         print(accFoldList)
        acc.append(accFoldList)

    for roundIndex in range(roundNum):
        roundAcc = []
        for foldIndex in range(foldNum):
            roundAcc.append(acc[foldIndex][roundIndex])
        meanAcc.append(np.mean(roundAcc))
        varAcc.append(np.sqrt(np.var(roundAcc)))

    return meanAcc, varAcc

randomAccFile = "./activeLearning_random_electronics_822109_acc.txt"
randomMeanAcc, randomVarAcc = readAccFile(randomAccFile)

marginAccFile = "./activeLearning_margin_electronics_8221010_acc.txt"
marginMeanAcc, marginVarAcc = readAccFile(marginAccFile)

CBAccFile = "./activeLearning_CB_electronics_822104_acc.txt"
CBMeanAcc, CBVarAcc = readAccFile(CBAccFile)

LUCBAccFile = "./activeLearning_LUCB_electronics_8221020_acc.txt"
LUCBMeanAcc, LUCBVarAcc = readAccFile(LUCBAccFile)

transferALAccFile = "./transferALelectronics_821233.txt"
transferALMeanAcc, transferALVarAcc = readAccFile(transferALAccFile)

# plt.figure(figsize=[10, 5])
# plt.figure()

x = [i for i in range(3, 100)]

ax.plot(x, randomMeanAcc[3:100], label="random learning ")

ax.plot(x, marginMeanAcc[3:100], label="margin active learning ")

ax.plot(x, CBMeanAcc[3:100], label="CB active learning")

ax.plot(x, LUCBMeanAcc[3:100], label="LUCB active learning")

ax.plot(x, transferALMeanAcc[3:100], label="transfer AL margin")

plt.xlabel("active iterations")
plt.ylabel("accuracy")
plt.title("active learning")
plt.legend(loc="lower right")
plt.savefig("activeLearning_electronics.pdf")
# plt.show()