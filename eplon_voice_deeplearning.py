import numpy as np
from tqdm import tqdm
import glob
import sys
import eplon_voice_net1
import eplon_voice_net2

DIR_PREDICT_OUTPUT = "./predict_441kHz_output"

#ペリオドグラム用
def eplon_voice_deeplearning_input_learning_data(pathname):

    file_num = len(glob.glob(pathname+"/data/*"))
    print("total file numbers: ",file_num)

    data_X = np.zeros([file_num, 28, 28])
    data_y = np.zeros(file_num)

    for n in tqdm(range(file_num)):
        data_X[n,:,:]   = np.load(pathname+"/data/"+str(n+1)+".npy")
        data_y[n]       = np.load(pathname+"/label/"+str(n+1)+".npy")

    print("data_X shape:",data_X.shape)
    print("data_y shape:",data_y.shape)

    l = list(zip(data_X, data_y))
    np.random.shuffle(l)
    data_X_s, data_y_s = zip(*l)

    print("data_X_s shape:",np.array(data_X_s).shape)
    print("data_y_s shape:",np.array(data_y_s).shape)

    rate_n = int(np.array(data_X_s).shape[0]*90/100)
    print("rate_num :",rate_n)
    return (np.array(data_X_s[:rate_n]), np.array(data_y_s[:rate_n])),(np.array(data_X_s[rate_n:]), np.array(data_y_s[rate_n:]))
