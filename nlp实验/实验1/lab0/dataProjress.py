'''
@Author: jiawangwang
@Time  : 2022/10/6
符号                                代号
Initial_probability                 0
Emission_probability                1
Transfer_probability                2
pinyin2hanzi                        3
'''
import  numpy as np
import os
import re
import math
import json
import pypinyin
import string

'''训练得到汉语词典'''
def train3(filepath):
    f = open(filepath, encoding='utf-8')
    print("开始构造汉语词典文件")
    pinyin2hanzi={}
    # 按行读取
    for line in f.readlines():
        # 将每行拼音与汉字之间的零宽不换行空格换为普通空格
        line = re.sub(r'[\ufeff]', '', line)
        # 将每行按空格切分并放入line列表中，一共有两个部分，
        # 其中line[0]为拼音，line[1]为对应的汉字（一堆）
        line = line.strip().split()
        # 存入汉语字典中
        pinyin2hanzi[line[0]] = line[1]
    f.close()
    print("汉语词典构造完成")
    return pinyin2hanzi



'''训练得到初始概率和转移概率'''
'''并分别存在两个不同的文件中'''
def train02(filename):
    print("开始初始概率和转移概率文件")
    '''
    :param filename:
    :return:
    '''
    #初始矩阵
    Initial_probability={}
    #转移矩阵
    Transfer_probability={}


    f=open(filename,encoding='utf-8')
    singleWord={}        #存放单个词的频数
    doubleWords={}       #存放两个词一起的频数
    #一共有多少个句子
    num=0
    for line in f.readlines():
        # 匹配所有中文形成一个list包,按符号分割出所有的字符
        line1 = re.findall('[\u4e00-\u9fa5]+', line)
        for words in line1:
            #存前一个字符
            pre=' '
            #计算一个字的频数和两个字连续出现的频数
            for word in words:
                if word in singleWord:
                    singleWord[word]+=1
                else:
                    singleWord[word]=1
                if pre != ' ':
                    if  pre+word in doubleWords:
                        doubleWords[pre+word]+=1
                    else:
                        doubleWords[pre+word]=1

                pre=word
            num+=1
    f.close()

    #求初始概率:
    for i in singleWord.keys():
        Initial_probability[i]=singleWord[i]/num
    #保存初始概率文件
    np.save("data/Initial_probability",Initial_probability)
    print('Initial_probability.npy已经保存在data文件下')
    # 求转移概率
    for i in doubleWords.keys():
        Transfer_probability[i] = doubleWords[i] / singleWord[i[0]]
    # 保存转移概率文件
    np.save("data/Transfer_probability", Transfer_probability)
    print('Transfer_probability.npy已经保存在data文件下')

'''发射概率文件'''
def train1(filename):
    print("开始构造发射概率文件")
    f = open(filename, encoding='utf-8')
    #发射概率列表
    Emission_probability={}
    for line in f.readlines():
        # 匹配所有中文形成一个list包
        line1 = re.findall('[\u4e00-\u9fa5]+', line)
        for words in line1:
            #汉字转成拼音,转化之后为一个由拼音组成的列表
            ans = pypinyin.lazy_pinyin(words)
            for i in range(len(ans)):
                if ans[i] not in Emission_probability:
                    Emission_probability[ans[i]]={}
                    Emission_probability[ans[i]][words[i]] = 1
                else:
                    if words[i] not in Emission_probability[ans[i]]:
                        Emission_probability[ans[i]][words[i]]=1
                    else:
                        Emission_probability[ans[i]][words[i]] += 1
    f.close()
    for key in Emission_probability:
        s=sum(Emission_probability[key].values())
        for key1 in Emission_probability[key]:
            Emission_probability[key][key1]=Emission_probability[key][key1]/s

    # 保存发射概率文件
    np.save("data/Emission_probability", Emission_probability)
    print('Emission_probability.npy已经保存在data文件下')





