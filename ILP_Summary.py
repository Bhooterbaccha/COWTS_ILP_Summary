from __future__ import print_function
from optlang.glpk_interface import Model, Variable, Constraint, Objective
from sympy import sympify

import sys
import os
from operator import itemgetter
#from gurobipy import *
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
for x in range(0,1):
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

        optlang_solve = open("optlang_solve.py", 'w')
        optlang_solve.write("from __future__ import print_function\n")
        optlang_solve.write("from optlang.glpk_interface import Model, Variable, Constraint, Objective\n\n")

        m = Model(name='sol1')

        ############ First Add tweet variables ############################################################
        
        sen_var = []
        for i in range(0,len(sen),1):
            sen_var.append("x%d" % (i+1))
            #sen_var[i] = Variable("x%d" % (i+1), lb=0)
            #print(sen_var[i])
            #print("x%d" % (i+1))
            optlang_solve.write("x%d" % (i+1) + " = Variable('" + "x%d" % (i+1) + "', lb=0)"  "\n")

        

        ############ Add entities variables ################################################################

        con_var = []
        for i in range(0,len(entities),1):
            con_var.append("y%d" % (i+1))
            #con_var[i] = Variable("y%d" % (i+1), lb=0)
            #print(con_var[i])
            optlang_solve.write("y%d" % (i+1) + " = Variable('" + "y%d" % (i+1) + "', lb=0)"  "\n")

        
        
        ########### Integrate Variables ####################################################################
        #m.update()

        #print sen_var

        #print con_var

        #p = "" # Contains objective function
        p_x = []
        for i in range(0,len(sen),1):
            p_x.append("x%d" % (i+1))
            p_x[i] = "x%d" % (i+1)
            #p += p_x[i] + ' + '
            #p = str(' + '.join(p_x))
        #print(p)

        p_y = []
        for i in range(0,len(entities),1):
            p_y.append("y%d" % (i+1))
            p_y[i] = str(word[entities[i]]) + ' * ' + "y%d" % (i+1)
            #p = p + str(' + '.join(p_y)

        p = p_x + p_y
        #print(p)

        #print(' + '.join(p))
        p_obj = (' + '.join(p))
        #obj = Objective(sympify(p_obj), direction='max')

        #C1 = LinExpr()  # Summary Length constraint

        c1 = ""
        c1_x = []
        counter = -1

        #c = ""
        #c_y = []
        constr = []

        for i in range(0,len(sen),1):
            c1_x.append("x%d" % (i+1))
            c1_x[i] = "x%d" % (i+1)
            #c1 += c1_x[i] + ' + '
            v = tweet_word[i+1][1]
            c = ""
            c_y = []
            flag = 0
            entities = list(entities)
            #print(len(entities))
            for j in range(0,len(entities),1):
                if entities[j] in v:
                    #print(entities[j])
                    flag+=1
                    c_y.append("y%d" % (j+1))
                    #c_y[j] = "y%d" % (j+1)
                    #c += c_y[j] + ' + '
            #print(' + '.join(c_y))

                    #c_y[j] = str(con_var[j]) + '*' + "y%d" % (j+1)
            if flag>0:
                counter+=1
                constr.append("c%d" % (i+1))
                #constr[i] = Constraint(sympify(' + '.join(c_y)), lb=flag * sen_var[i])
                #constr[i] = str(' + '.join(c_y)) + ' + ' + str(flag * sen_var[i])
                #constr[i] = Constraint(sympify(str(' + '.join(c_y)) + ' - ' + str(flag * sen_var[i])), lb=0)
                #print("c%d" % (i+1) + ': ' + str(' + '.join(c_y)) + ' - ' + str(flag * sen_var[i]))
                optlang_solve.write("c%d" % (counter+1) + " = Constraint(" + ' + '.join(c_y) + " - " + "%d * " % flag + sen_var[i] + ", lb=0)"  "\n")
                # counter can be replaced by i

        #print(c1)
        # for i in range(0,len(sen),1):
        #     #P += sen_var[i]
        #     C1 += sen_var[i]
        #     v = tweet_word[i+1][1]
        #     #print(v)
        #     C = LinExpr()
        #     flag = 0
        #     entities = list(entities)       
        #     for j in range(0,len(entities),1):
        #         if entities[j] in v:
        #             flag+=1
        #             C += con_var[j]
        #     if flag>0:
        #         counter+=1
        #         m.addConstr(C, GRB.GREATER_EQUAL, flag * sen_var[i], "c%d" % (counter))
        
        constr1 = []

        for i in range(0,len(entities),1):
            flag = 0
            c_x_en = []
            for j in range(0,len(sen),1):
                v = tweet_word[j+1][1]
                if entities[i] in v:
                    flag = 1
                    c_x_en.append("x%d" % (j+1))

            #print(c_x_en)
            if flag==1:
                counter+=1
                #constr1.append("c_%d" % (i+1))
                constr1.append("c%d" % (i+1+len(sen)))
                #constr1[i] = Constraint(sympify(str(' + '.join(c_y)) + ' - ' + str(flag * sen_var[i])), lb=0)
                #print("c_%d" % (i+1) + ': ' + str(' + '.join(c_x_en)) + ' * ' + str(con_var[i]))
                optlang_solve.write("c%d" % (counter+1) + " = Constraint(" + ' + '.join(c_x_en) + " - " + con_var[i] + ", lb=0)" "\n")
                # counter can be replaced with i+len(sen)

        # for i in range(0,len(entities),1):
        #     #P += word[entities[i]] * con_var[i]
        #     C = LinExpr()
        #     flag = 0
        #     for j in range(0,len(sen),1):
        #         v = tweet_word[j+1][1]
        #         if entities[i] in v:
        #             flag = 1
        #             C += sen_var[j]
        #     if flag==1:
        #         counter+=1
        #         m.addConstr(C,GRB.GREATER_EQUAL,con_var[i], "c%d" % (counter))

        counter+=1
        optlang_solve.write("c%d" % (counter+1) + " = Constraint(" + ' + '.join(c1_x) + ", ub=" + "%d" % L + ")" "\n")

        
        #m.addConstr(C1,GRB.LESS_EQUAL,L, "c%d" % (counter))

        optlang_solve.write("\nobj" + " = Objective(" + ' + '.join(p) + ", direction='max')" "\n")
        constr2 = ["c%d" % (counter+1)]
        
        
        ################ Set Objective Function #################################
        #m.setObjective(P, GRB.MAXIMIZE)
        #m.write("output.lp")
        constr3 = constr + constr1 + constr2
        #print(constr)
        #print(constr1)
        #print(constr3)

        optlang_solve.write("\nmodel = Model(name='Simple model')\n")
        optlang_solve.write("model.objective = obj\n")
        optlang_solve.write("model.add([" + ', '.join(constr3) + "])\n")
        optlang_solve.write("status = model.optimize()\n")
        optlang_solve.write("print(\"status:\", model.status)\n")
        optlang_solve.write("print(\"objective value:\", model.objective.value)\n")
        optlang_solve.write("print(\"----------\")\n")
        #optlang_solve.write("for var_name, var in model.variables.iteritems():\n")
        #optlang_solve.write("\tprint(var_name, \"=\", var.primal)\n")

        ############### Set Constraints ##########################################

        fo = open(ofname,'w')
        #m.write(out.mps)
        #m.write(out.lp)
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
