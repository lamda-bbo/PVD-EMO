# Run all the cells up to the Experiments Heading to load the algorithm and sample data
# The subsequent cell generates the designs that are needed to run the experiments, which will
# be dumped into the "outputs" folder.
import os
import pickle
from xmlrpc.client import boolean
import numpy as np
import pandas as pd
import random
import gzip
from copy import deepcopy
from matplotlib import pyplot as plt
import argparse
import torch
import MOEA as moea
import dill


def load_data(input_file, times, ea, flag=None):
    if flag == 'ifexists':
        if not os.path.isfile(input_file +'/'+'time_'+str(times)+'.pkl'):
            return {}
    if ea == 'nsga2_o':
        with open(input_file +'/'+'time_'+str(times)+'.pkl','rb') as f:
            data = dill.load(f)
    else:
        with open(input_file +'/'+'time_'+str(times)+'.pkl','rb') as f:
            data=pickle.load(f, encoding='iso-8859-1')
    return data


# Utility functions

# f(x) = min(n, x)
def getThresholdUtility(n):
    a=torch.tensor(np.arange(0,n+1,1),dtype = torch.float64)
    return a

def main(args):
    # save args settings
    data_name=args.dataFile
    ea=args.ea
    times=args.times
    T=args.T
    # threshold=args.threshold
    pc=args.pc
    k=args.k
    threshold=int(k*args.threshold_ratio)
    cm = args.continueMOEA
    print(args.dataFile, args.ea, args.T, args.k, int(k*args.threshold_ratio))

    # read data
    with open("data/{}.pkl".format(data_name), 'rb') as fin:
        data = pickle.load(fin)

    # save algorithm settings
    res_file='result/'+ data_name+ '_'+ ea +'_' + str(k) + '_' + str(threshold) + '_' + str(pc) #+ '/times_' + str(times)
    
    if not os.path.exists(res_file):
        os.makedirs(res_file)
    res_pkl = load_data(res_file, times, ea, 'ifexists')
    param = {'data': data, 'k': k, 'threshold': args.similar, 'utility': getThresholdUtility(threshold),
            'res_file':res_file, 'times':times,'ea':ea,'pc':pc ,'res_pkl':res_pkl}

    algo = moea.MOEA(param)
    algo.setEvaluateTime(T)
    algo.run_MOEA(cm)
   

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-dataFile',type=str,default= "mhc1_credences")#'mhc1_credences', 'mhc2_credences', 'mhc1_binarized','mhc2_binarized'
    argparser.add_argument('-similar',type=int, default=6)
    argparser.add_argument('-ea',type=str, default="PVD-GSEMO-WR") #'PVD-GSEMO-WR', 'PVD-GSEMO-R', 'PVD-GSEMO', 'PVD-NSGA-II-WR'
    argparser.add_argument('-pc', type=float, help="probability of crossover", default=0.9)
    argparser.add_argument('-threshold_ratio', type=float, default=0.25) #'0.25'
    argparser.add_argument('-k', type=int, default=40)
    argparser.add_argument('-times', type=int, default=0)
    argparser.add_argument('-T', type=int, default=25)
    argparser.add_argument('-continueMOEA', type=bool, default=False)

    args = argparser.parse_args()
    main(args)
