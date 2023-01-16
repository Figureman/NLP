import re
import os
import numpy as np
import pypinyin
import json
'''HMM模型'''
class HMM:
    '''初始化函数'''
    def __init__(self,Initial_probability,Emission_probability,Transfer_probability,pinyin2hanzi):
        self.Initial_probability=Initial_probability            #初始概率  0
        self.Emission_probability=Emission_probability          #发射概率  1
        self.Transfer_probability=Transfer_probability          #转移概率  2
        self.pinyin2hanzi=pinyin2hanzi                          #汉语字典  3

    '''维特比算法'''

    def viterbi(self, word_list, pinyin_list, n, id2word, label_list, ):
        """
        :param word_list:   每个拼音对应的隐藏状态矩阵
        :param n:   可能观察到的状态数， 对应为汉字数量
        :param id2word:    id到汉字的映射
        :label_list:    原句
        :return:
        """
        T = len(word_list)  # 观察状态的长度

        delta = np.zeros((T, n))
        # 保存转移下标值
        psi = np.zeros((T, n), dtype=int)

        # 初始化第一个字符的转移概率， 设置为每个词在词典中的单独出现的概率
        words = word_list[0]
        for w in words:
            if id2word[w] not in self.Initial_probability:
                delta[0][w] = 0
            else:
                delta[0][w] = self.Initial_probability[id2word[w]]
                # 动态规划计算
        for idx in range(1, T):
            words = word_list[idx]
            for i in range(len(words)):
                max_value = 0
                pre_words = word_list[idx - 1]
                index = 0
                for j in range(len(pre_words)):
                    tmp_key = id2word[pre_words[j]] + id2word[words[i]]
                    # 获得转移概率，如果不存在，转移概率则为0
                    if tmp_key in self.Transfer_probability:
                        prob = self.Transfer_probability[tmp_key]
                    else:
                        prob = 0

                    tmp_value = delta[idx - 1][pre_words[j]] * prob

                    if max_value < tmp_value:
                        max_value = tmp_value
                        index = j
                if pinyin_list[idx] not in self.Emission_probability:
                    emit_prob = 0
                elif id2word[words[i]] not in self.Emission_probability[pinyin_list[idx]]:
                    emit_prob = 0
                else:
                    emit_prob = self.Emission_probability[pinyin_list[idx]][id2word[words[i]]] * max_value

                delta[idx][words[i]] = emit_prob
                psi[idx][words[i]] = pre_words[index]

        prob = 0
        path = np.zeros(T, dtype=int)
        path[T - 1] = 1
        # 获取最大的转移值
        for i in range(n):
            if prob < delta[T - 1][i]:
                prob = delta[T - 1][i]
                path[T - 1] = i

                # 最优路径回溯
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1][path[t + 1]]

            # 生成解析结果
        final_word = ""
        a = 0
        for i in range(T):
            final_word += id2word[path[i]]
            if id2word[path[i]] == label_list[i]:
                a += 1
        print('原句：',label_list)
        print('转换：', final_word)
        print('准确率：', a * 1.0 / T)
    #输入英文数据和对应标签
    def dataDetail(self,yinwen,label):
        # 找每个拼音可能对应的汉字集合
        pinyin_list=yinwen.lower().split()
        word_list = []
        for i in pinyin_list:
            temp = re.findall(r'[\u4e00-\u9fa5]', self.pinyin2hanzi[i])
            word_list.append(temp)
        # 为每一个字对应一个id，每一个id对应一个字
        words = set()
        for wl in word_list:
            for w in wl:
                words.add(w)
        word2idx = dict()
        id2word = dict()
        idx = 0
        for w in words:
            word2idx[w] = idx
            id2word[idx] = w
            idx += 1

        word_id_list = [None] * len(word_list)
        for i, wl in enumerate(word_list):
            word_id_list[i] = [None] * len(wl)
            for j, w in enumerate(wl):
                word_id_list[i][j] = (word2idx[w])
        #开始训练
        self.viterbi(word_id_list, pinyin_list, len(words), id2word, label )
    #输入测试集函数
    def test(self, test_path, ):
        f = open(test_path, encoding='gb2312')
        lines = f.readlines()
        flag = 1
        pinyin = []
        label = []
        #分割拼音和汉字
        for line in lines:
            if flag == 1:
                pinyin.append(line)
                flag = 0
            else:
                label.append(line)
                flag = 1
        for i in range(len(pinyin)):
            self.dataDetail(pinyin[i],label[i])



