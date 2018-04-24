import pandas as pd
import numpy as np

def load_data(filename):
    df = pd.read_csv(filename, delimiter="\n")
    data = df.values[:,0]
    texts = []
    labels = []
    count = 10 * [0]
    for i in range(data.shape[0]):
        seq = data[i][5:]
        # print(data[i])
        # print(i)
        label = int(data[i][0])-1
        count[label] = count[label]+1
        if (label >= 3):
            continue
        texts.append(seq)
        labels.append(label)
    #print(count)
    return texts, np.array(labels)


#load_data("data/train.txt")

