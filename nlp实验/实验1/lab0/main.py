import numpy as np
from dataProjress import *
from HMM import  HMM
if __name__=='__main__':

    '''三大训练模型,只需要第一次调用即可'''
    #train02('data/toutiao_cat_data.txt')
    #train1('data/toutiao_cat_data.txt')

    '''模型构建'''

    pinyin2hanzi=train3('data\pinyin2hanzi.txt')
    Initial_probability=np.load('data/Initial_probability.npy',allow_pickle='TRUE').item()
    Emission_probability=np.load('data/Emission_probability.npy',allow_pickle='TRUE').item()
    Transfer_probability=np.load('data/Transfer_probability.npy',allow_pickle='TRUE').item()
    '''模型参数渗入'''
    model=HMM(Initial_probability,Emission_probability,Transfer_probability,pinyin2hanzi)
    model.dataDetail('lao shi hao wo shi zhao jia le','老师好我是赵家乐')
    #print(model.test('data/测试集.txt'))

