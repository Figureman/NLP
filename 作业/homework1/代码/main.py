import  numpy as np
import os
import re
import math
import json
import pypinyin
import string

#构建数据库
doubleWords = {}
def train02(filename):
    doubleWords_pre={}
    f=open(filename,encoding='utf-8')
    #一共有多少个句子
    num=0
    for line in f.readlines():
        # 匹配所有中文形成一个list包,按符号分割出所有的字符
        line1 = re.findall('[\u4e00-\u9fa5]+', line)
        for words in line1:
            #存前一个字符
            words1='B'+words+'E'
            pre=' '
            #计算一个字的频数和两个字连续出现的频数
            for word in words1:
                if pre != ' ':
                    if pre in doubleWords_pre:
                        if word in doubleWords_pre[pre]:
                            doubleWords_pre[pre][word]+=1
                        else:
                            doubleWords_pre[pre][word]=1
                    else:
                        doubleWords_pre[pre]={}
                pre=word
            num+=1
    f.close()
    # 保存初始概率文件
    np.save("data/doubleWords_pre", doubleWords_pre)

#采用平滑处理方法+二元文法

def fun1(worda ,wordb):
    if worda in doubleWords:
        if wordb in doubleWords[worda]:
            sum=0
        else:
            doubleWords[worda][wordb]=0
    else:
        doubleWords[worda]={}
        doubleWords[worda][wordb]=0
    fenzi=doubleWords[worda][wordb]+1
    fenmu=0
    for key in doubleWords[worda]:
        fenmu=fenmu+doubleWords[worda][key]+1
    doubleWords[worda][wordb]+=1
    return (fenzi/fenmu)

#采用二元文法计算概率,需要自己加'B'和'E':
def fun(word):
    sum=1.0
    for i in range(len(word)-1) :
        sum=sum*fun1(word[i],word[i+1])
    return sum

#预测结果
def predict(word):
    preRes={}
    str=word[len(word)-1]
    s = sum(doubleWords[str].values())
    for key in doubleWords[str]:
        if key=='E':continue
        str1=str+key
        preRes[key]=fun(str1)
    return preRes

#将预测结果排序，选取前五个
def Paixu(word):
    Result=[]
    preRes=predict(word)
    preRes = sorted(preRes.items(), key=lambda x: x[1], reverse=True)
    for i in range(5):
        Result.append(preRes[i][0])
    return Result

#二元文法中后一个字出现的概率仅仅取决于前一个字
if __name__ == '__main__':
    #下一行只需要在更换数据集时运行即可。
    #train02('data/sentiment_analysis_trainingset.txt')
    doubleWords=np.load('data/doubleWords_pre.npy',allow_pickle='TRUE').item()
    while(True):
        word = input("请输入您要测试的句子\n")
        if word=='q':
            print("结束")
        result = Paixu(word)
        print("最可能的五个词，它们的可能性从大到小分别是:")
        print(result)

