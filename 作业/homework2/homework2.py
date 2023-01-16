import re
import jieba
import  numpy as np
import copy
#构建数据库
wordset={}
diction={}
def train02(filename):
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
            words1List=jieba.lcut(words1)
            #计算一个词的频数和两个词连续出现的频数
            for word in words1List:
                for word in words1List:
                    if word in diction.keys():
                        diction[word] += 1
                    else:
                        diction[word] = 1
                if pre != ' ':
                    if pre in wordset:
                        if word in wordset[pre]:
                            wordset[pre][word]+=1
                        else:
                            wordset[pre][word]=1
                    else:
                        wordset[pre] = {}
                        wordset[pre][word] = 1
                pre=word
            num+=1
    f.close()
    np.save("./data/wordset.npy", wordset)
    np.save("./data/dict.npy", diction)

class ChineseTokenizer:
    def __init__(self,dictPath,diction,wordset):
        """
        中文分词器，包括FMM与BMM算法
        
        """
        #统计词的词典
        self.dictionary = self.createDictionary(dictPath)
        #统计一个词出现的频率
        self.diction=diction
        #统计两个词一起出现的频率
        self.wordset=wordset
    
    #创造统计词的词典
    def createDictionary(self,dictPath):
        Dictonary = list()
        for line in open(dictPath,encoding='utf-8'):
            Dictonary += line.strip().split(' ')
        return Dictonary

    def FMM(self, sentence):
        """
        正向最大匹配算法，正向移动，缩短子串时删除末尾的字
        param sentence: 待分词句子
        return res: 分词列表
        """
        res = []
        # 外层循环切句子
        while len(sentence) > 0:
            # 初始化一个切词划分窗口，取词典最长词长度和句长的最小值
            split_win_len = min(4, len(sentence))
            # 初始化词
            sub_sen = sentence[0:split_win_len] # 正向移动
            # 内层循环匹配词
            while len(sub_sen) > 0:
                # 如果词在词典中
                if sub_sen in self.dictionary:
                    res.append(sub_sen)
                    break
                # 如果词长度为1，说明词典中没有当前词
                elif len(sub_sen) == 1:
                    res.append(sub_sen)
                    break
                # 如果当前词在词典中没有，那么删除末尾的字，再进入while匹配
                else:
                    split_win_len -= 1 # 划分窗口缩短
                    sub_sen = sub_sen[0:split_win_len] # 子串尾部pop
            # 一次匹配结束更新原句子
            sentence = sentence[split_win_len:]
        return res


    def BMM(self, sentence):
        """
        逆向最大匹配算法，逆向移动，缩短子串时删除首部的字
        
        param :
        return:
        """
        res = []
        while len(sentence) > 0:
            split_win_len = min(4, len(sentence))
            sub_sen = sentence[-split_win_len:]
            while len(sub_sen) > 0:
                if sub_sen in self.dictionary: # 逆向移动
                    res.append(sub_sen)
                    break
                elif len(sub_sen) == 1:
                    res.append(sub_sen)
                    break
                else:
                    split_win_len -= 1 # 划分窗口缩短
                    sub_sen = sub_sen[-split_win_len:] # 子串首部pop
            sentence = sentence[0:-split_win_len]
        res = res[::-1]
        return res

    #查找wordset
    def findWordset(self,worda,wordb):
        if worda in self.wordset:
            if wordb in self.wordset[worda]:
                return self.wordset[worda][wordb]
            else:
                return 0
        else:
            return 0
    
    #查找wordset中某一个的例子
    def findWordset1(self,word):
        if word in self.wordset:
            return len(self.wordset[word])
        else:
            return 1

    #查找diction
    def findDiction(self,word):
        if word in self.diction:
            return self.diction[word]
        else:
            return 0

    #计算得分，用拟二元文法模型

    def pingfen(self,Res):
        Result=copy.deepcopy(Res)
        P = 1
        Result.append('E')
        Result.insert(0,'B')
        for i in range(1,len(Result)):
            res=1.0*(self.findWordset(Result[i-1],Result[i])+1)/(self.findDiction(Result[i-1])+self.findWordset1(Result[i-1]))
            P=P*res
        return P


    
if __name__=='__main__':
    #train02('C:/Users/19033/Desktop/PPT/NLP/作业/homework2/data/news2.txt')
    diction = np.load('F:/课件/PPT/NLP/作业/homework2/data/dict1.npy', allow_pickle='TRUE').item()
    wordset = np.load('F:/课件/PPT/NLP/作业/homework2/data/wordset.npy', allow_pickle='TRUE').item()
    #print(diction)
    #print(wordset)
    model=ChineseTokenizer('F:/课件/PPT/NLP/作业/homework2/data/dict.txt',diction,wordset)
    sentences="全国人民代表大会在北京人民大会堂隆重召开"
    Fmmlist=model.FMM(sentences)
    Bmmlist=model.BMM(sentences)
    FmmlistPingfen=model.pingfen(Fmmlist)
    BmmlistPingfen=model.pingfen(Bmmlist)
    print("前向分词结果为:",Fmmlist)
    print("前向分词概率为:",FmmlistPingfen)
    print("后向分词结果为:",Bmmlist)
    print("后向分词概率为:",BmmlistPingfen)   
    #消除歧义
    if(FmmlistPingfen>BmmlistPingfen):
        print("我们选择前向分词,即最终分词结果为:")
        print(Fmmlist)
    else:
        print("我们选择后向分词,即最终分词结果为:")
        print(Bmmlist)


    
