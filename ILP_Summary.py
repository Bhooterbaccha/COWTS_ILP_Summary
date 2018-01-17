import sys
import os
from operator import itemgetter
from gurobipy import *
import math
from textblob import *
import re
import time
import times
import codecs
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import itertools
start_time = time.time()
lmtzr = WordNetLemmatizer()

# IMPLEMENTATION BY KOUSTAV RUDRA, IIT KHARAGPUR
for x in xrange(0,1):
    pass
    WORD = re.compile(r'\w+')
    cachedstopwords = stopwords.words("english")
    Tagger_Path = './cmu'  # NEED TO FILL IN THE PATH TO CMU TAGGER
    AUX = ['be','can','could','am','has','had','is','are','may','might','dare','do','did','have','must','need','ought','shall','should','will','would','shud','cud','don\'t','shouldn\'t','wouldn\'t','couldn\'t','didn\'t','doesn\'t']

    def summarize(ifname,ofname,L):
        
        tag = ['$','V','N']
        TAGREJECT = ['#','@','~','U','E','G',',']

        ''' POS Tagging the tweets '''

        command = 'sh ./cmu/runTagger.sh --output-format conll ' + ifname + ' > tagfile.txt'
        os.system(command)
        ft = open(ifname,'r')
        fp = open('tagfile.txt','r')
        tweet_dic = {}
        tweet_index = 0
        count = 0
        temp = set([])
        
        for l in fp:    
            wl = l.split()
            if len(wl)>1:
                if wl[1]=='$':
                    s = wl[0].strip(' \t\n\r')
                    try:
                        w = str(numToWord(int(s)))
                        if len(w.split()) > 1:
                            w = s
                    except Exception as e:
                        w = str(s)
                    if len(w.lower())>1:
                        temp.add(w.lower())
                elif wl[1]=='N':
                    s = wl[0].strip(' \t\n\r').lower()
                    try:
                        w = lmtzr.lemmatize(s)
                    except Exception as e:
                        w = s
                    if len(w.lower())>1:
                        temp.add(w.lower())
                elif wl[1]=='V':
                    s = wl[0].strip(' \t\n\r').lower()
                    try:
                        w = Word(s)
                        x = w.lemmatize("v")
                    except Exception as e:
                        x = s
                    if x.lower() not in AUX:
                        if len(x.lower())>1:
                            temp.add(x.lower())
                else:
                    pass
            else:
                tweet_dic[tweet_index] = [str(tweet_index),ft.readline().strip(' \t\n\r'),temp]
                temp = set([])
                tweet_index+=1
        fp.close()
        ft.close()
            
        TWEET = []
        for k,v in tweet_dic.items():
            t = (k,v[0],v[1],v[2])
            TWEET.append(t)

        TWEET.sort(key=itemgetter(0))


        ''' RunTime Summarization '''

        tag = ['$','V','N']
        
        ''' dictionaries which contains details about content words '''
        word = {}

        ''' dictionary contain tweets of current window we want to summarize '''
        tweet_cur_window = {}

        t0 = time.time()

        for i in range(0,len(TWEET),1):
            t = TWEET[i]

            ''' This dictionary is for current window which does not appear earlier '''
            for w in t[3]:
            
                if word.__contains__(w)==True:
                    v = word[w]
                    word[w] = v+1
                else:
                    word[w] = 1
            
            tweet_cur_window[t[1]] = [t[2],0,t[3]]
            if i+1==len(TWEET):

                print('Prepare Summary at: ' + str(i+1))

                ######################## Compute tf-idf Score of Content Words ######################
                    
                weight = {}
                weight = compute_tf(word)
                    
                ##################### Start Local Summarization #####################################

                optimize(tweet_cur_window,weight,ofname,L)
                
                t1 = time.time()
                print('Tweet tick: ' + str(i+1) + ' Time Difference: ' + str(t1-t0))
        
        fp.close()

    def optimize(tweet,weight,ofname,L):

        ################################ Extract Tweets and Content Words ##############################
        word = {}
        tweet_word = {}
        tweet_index = 1
        for  k,v in tweet.items():
            set_of_words = v[2]
            for x in set_of_words:
                if word.__contains__(x)==False:
                    if weight.__contains__(x)==True:
                        p1 = round(weight[x],4)
                    else:
                        p1 = 0.0
                    word[x] = p1
            tweet_word[tweet_index] = [v[1],set_of_words,v[0]]
            tweet_index+=1

        ############################### Make a List of Tweets ###########################################
        sen = tweet_word.keys()
        sen = sorted(sen)
        entities = list(word.keys())

        ################### Define the Model #############################################################

        m = Model("sol1")

        ############ First Add tweet variables ############################################################
        
        sen_var = []
        for i in range(0,len(sen),1):
            sen_var.append(m.addVar(vtype=GRB.BINARY, name="x%d" % (i+1)))

        ############ Add entities variables ################################################################

        con_var = []
        for i in range(0,len(entities),1):
            con_var.append(m.addVar(vtype=GRB.BINARY, name="y%d" % (i+1)))
        
        ########### Integrate Variables ####################################################################
        m.update()

        P = LinExpr() # Contains objective function
        C1 = LinExpr()  # Summary Length constraint
        counter = -1
        for i in range(0,len(sen),1):
            P += sen_var[i]
            C1 += sen_var[i]
            v = tweet_word[i+1][1]
            #print(v)
            C = LinExpr()
            flag = 0
            entities = list(entities)       
            for j in range(0,len(entities),1):
                if entities[j] in v:
                    flag+=1
                    C += con_var[j]
            if flag>0:
                counter+=1
                m.addConstr(C, GRB.GREATER_EQUAL, flag * sen_var[i], "c%d" % (counter))
        
        for i in range(0,len(entities),1):
            P += word[entities[i]] * con_var[i]
            C = LinExpr()
            flag = 0
            for j in range(0,len(sen),1):
                v = tweet_word[j+1][1]
                if entities[i] in v:
                    flag = 1
                    C += sen_var[j]
            if flag==1:
                counter+=1
                m.addConstr(C,GRB.GREATER_EQUAL,con_var[i], "c%d" % (counter))

        counter+=1
        m.addConstr(C1,GRB.LESS_EQUAL,L, "c%d" % (counter))
        
        ################ Set Objective Function #################################
        m.setObjective(P, GRB.MAXIMIZE)

        ############### Set Constraints ##########################################

        fo = open(ofname,'w')
        try:
            m.optimize()
            for v in m.getVars():
                if v.x==1:
                    temp = v.varName.split('x')
                    if len(temp)==2:
                        fo.write(tweet_word[int(temp[1])][2])
                        fo.write('\n')
        except GurobiError as e:
            print(e)
            sys.exit(0)
        fo.close()

    def compute_tf(word):
        
        score = {}
        for k,v in word.items():
            tf = 1 + math.log(v,2)
            score[k] = tf
        return score

    def numToWord(number):
        word = []
        if number < 0 or number > 999999:
                return number
                # raise ValueError("You must type a number between 0 and 999999")
        ones = ["","one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen"]
        if number == 0: return "zero"
        if number > 9 and number < 20:
                return ones[number]
        tens = ["","ten","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]
        word.append(ones[int(str(number)[-1])])
        if number >= 10:
                word.append(tens[int(str(number)[-2])])
        if number >= 100:
                word.append("hundred")
                word.append(ones[int(str(number)[-3])])
        if number >= 1000 and number < 1000000:
                word.append("thousand")
                word.append(numToWord(int(str(number)[:-3])))
        for i,value in enumerate(word):
                if value == '':
                        word.pop(i)
        return ' '.join(word[::-1])

    def main():
        try:
            _,L,ifname,ofname = sys.argv
        except Exception as e:
            print(e)
            sys.exit(0)
        summarize(ifname,ofname,int(L))
        print('Done')

    if __name__=='__main__':
        main()
    #print("--- %s seconds --- for ILP" % (time.time() - start_time))
print("--- %s seconds --- for ILP" % (time.time() - start_time))    
