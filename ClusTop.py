from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import nltk
import re
import pandas as pd
from sklearn.metrics import accuracy_score
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import webbrowser

main = tkinter.Tk()
main.title("ClusTop: A Clustering-based Topic Modelling Algorithm for Twitter using Word Networks")
main.geometry("1300x900")

global dataset
global filename
global wordnet
global precision, recall, fscore, pmi_score, tc_score
global target
global tweets, topic
global coword, bigram, clean

stopwords = set(stopwords.words('english'))
stemmer = PorterStemmer()
labels = ['bombing', 'earthquake', 'explosion', 'floods', 'hurricane', 'tornado']

def uploadDataset():
    global dataset, filename, target, tweets, topic
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset))
    text.insert(END,"\n\ndataset loaded")

def preprocessDataset():
    global dataset, filename, target, tweets, topic
    text.delete('1.0', END)
    dataset = dataset.values
    tweets = dataset[:,2]
    topic = dataset[:,4]
    for i in range(len(labels)):
        labels[i] = stemmer.stem(labels[i])
    target = []
    for i in range(len(topic)):
        lbl = stemmer.stem(topic[i].strip())
        target.append(labels.index(lbl))
    text.insert(END,"Dataset loaded\n\n")
    text.insert(END,"Preprocessing Completed\n")

#We have used this function to clean tweets by removing stop words such as "and the or what etc" and then perform stemming to get root words
#and then clean tweets    
def tweetsProcessing():
    global coword, clean
    for i in range(len(tweets)):
        tweet = tweets[i]
        tweet = tweet.replace("\n"," ")
        tweet = re.sub(r'[^a-zA-Z\s]+', '', tweet)
        tokens = word_tokenize(tweet.strip())
        words = ''
        for word in tokens:
            if len(word) > 2 and word not in stopwords:
                words+=stemmer.stem(word)+" "
        words+=stemmer.stem(topic[i])
        clean.append(words.strip())

#We have written this function to create a network with words and its weight
#words will be act like VERTEX and its weight value act like edges connection between two words which co occur        
def networkConstruction():
    global coword, clean
    for i in range(len(tweets)):
        tweet = word_tokenize(clean[i])
        word_pair = Counter(tweet)
        network_graph = {}
        for key, values in word_pair.items():
            if values >= 2 and key in network_graph.keys():#if graph already contains words vertex then append a edge
                network_graph[key].append(values/len(tweet))
            else:
                temp = []
                temp.append(values/len(tweet)) #calculating word weight
                network_graph[key] = temp
        coword[tweets[i]] = network_graph     


def cowordClusTop():
    global precision, recall, fscore, pmi_score, tc_score
    global coword, clean
    coword = {}
    clean = []
    text.delete('1.0', END)
    precision = []
    recall = []
    fscore = []
    pmi_score = []
    tc_score = []
    predict = np.zeros(len(target))
    text.insert(END,"Coword Topic Detection Output\n")
    text.insert(END,"========================================================================\n")
    if os.path.exists('model/unigram.txt'):
        with open('model/unigram.txt', 'rb') as file:
            coword = pickle.load(file)
        file.close()
    else:
        tweetsProcessing()
        networkConstruction()
        with open('model/unigram.txt', 'wb') as file:
            pickle.dump(coword, file)
        file.close()
    index = 0
    tc = 0
    pmi = 0
    #Louvian Algorithm
    for key, value in coword.items(): #find word pair and its neighbour with coword technique 
        #ranking/scoring will be done in ascending order 
        ranked = sorted(value.items(), key=lambda x: x[1], reverse = True)#ranked or score each word and its neighbour and then choose word with highest weight
        max_neighbor = ranked[0] #get the nieghbor with highest score
        max_score = max_neighbor[0]
        if max_score in labels:
            tname = labels.index(max_score)#word with high score with no more modularity will be detected as topic and all related topic assigned to same cluster
            if tname == target[index]:
                text.insert(END,"Tweet: "+key+"\n")
                text.insert(END,"Detected Topic: "+max_score+"\n\n")
                text.update_idletasks()
                predict[index] = tname
                tc = tc + max_neighbor[1][0]
                if len(ranked) > 1:
                    pmiValue = ranked[1]
                    pmi = pmi + pmiValue[1][0]
        else:
            for k, v in value.items():
                if k in labels:
                    tname = labels.index(k)
                    predict[index] = tname
                    tc = tc + max_neighbor[1][0]
                    if len(ranked) > 1:
                        pmiValue = ranked[1]
                        pmi = pmi + pmiValue[1][0]
        index += 1
    p = precision_score(target, predict,average='macro')
    r = recall_score(target, predict,average='macro')
    f = f1_score(target, predict,average='macro')
    pmi = pmi / len(coword)
    tc = tc / len(coword)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    pmi_score.append(pmi)
    tc_score.append(tc)
    text.insert(END,"ClusTop-Word-NA CO-WORD Precision: "+str(p)+"\n")
    text.insert(END,"ClusTop-Word-NA CO-WORD Recall   : "+str(r)+"\n")
    text.insert(END,"ClusTop-Word-NA CO-WORD FScore   : "+str(f)+"\n")
    text.insert(END,"ClusTop-Word-NA CO-WORD PMI      : "+str(pmi)+"\n")
    text.insert(END,"ClusTop-Word-NA CO-WORD TC       : "+str(tc)+"\n\n")


def biwordClusTop():
    text.delete('1.0', END)
    global precision, recall, fscore, pmi_score, tc_score
    global bigram, clean
    bigram = {}
    clean = []
    text.insert(END,"Bigram Topic Detection Output\n")
    text.insert(END,"========================================================================\n")
    predict = np.zeros(len(target))
    if os.path.exists('model/bigram.txt'):
        with open('model/bigram.txt', 'rb') as file:
            bigram = pickle.load(file)
        file.close()
    else:
        tweetsProcessing()
        networkConstruction()
        bigrams = nltk.word_tokenize(clean[0])  	
        bigram = list(nltk.bigrams(bigrams))
        words = ""
        for word in bigram: #here for bigram we will extract two words 0 and 1 with bigram
            words+=word[0]+" "+word[1]+" "
        words = nltk.word_tokenize(words.strip())
        with open('model/bigram.txt', 'wb') as file:
            pickle.dump(bigram, file)
        file.close()
    index = 0
    tc = 0
    pmi = 0
    #Louvian Algorithm
    for key, value in bigram.items():
        ranked = sorted(value.items(), key=lambda x: x[1], reverse = True)
        if len(ranked) > 0:
            max_neighbor = ranked[0]
            max_score = max_neighbor[0]
            if max_score in labels:
                tname = labels.index(max_score)
                if tname == target[index]:
                    text.insert(END,"Tweet: "+key+"\n")
                    text.insert(END,"Detected Topic: "+max_score+"\n\n")
                    text.update_idletasks()
                    predict[index] = tname
                    tc = tc + max_neighbor[1][0]
                    if len(ranked) > 1:
                        pmiValue = ranked[1]
                        pmi = pmi + pmiValue[1][0]
            else:
                for k, v in value.items():
                    if k in labels:
                        tname = labels.index(k)
                        if tname == target[index]:
                            predict[index] = tname
                            tc = tc + max_neighbor[1][0]
                            if len(ranked) > 1:
                                pmiValue = ranked[1]
                                pmi = pmi + pmiValue[1][0]
        index += 1
    p = precision_score(target, predict,average='macro')
    r = recall_score(target, predict,average='macro')
    f = f1_score(target, predict,average='macro')
    pmi = pmi / len(coword)
    tc = tc / len(coword)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    pmi_score.append(pmi)
    tc_score.append(tc)
    text.insert(END,"ClusTop-Word-NA Bigram Precision: "+str(p)+"\n")
    text.insert(END,"ClusTop-Word-NA Bigram Recall   : "+str(r)+"\n")
    text.insert(END,"ClusTop-Word-NA Bigram FScore   : "+str(f)+"\n")
    text.insert(END,"ClusTop-Word-NA Bigram PMI      : "+str(pmi)+"\n")
    text.insert(END,"ClusTop-Word-NA Bigram TC       : "+str(tc)+"\n\n")


def runLDA():
    text.delete('1.0', END)
    global precision, recall, fscore, pmi_score, tc_score, filename
    num_comp = 10
    iteration = 5
    method = 'online'
    offset = 50.
    state = 42
    minValue = 2
    maxValue = 0.95
    totalFeatures = 1000
    number_of_Words = 20
    dataset = pd.read_csv(filename)
    textData = dataset['tweet']
    vector = CountVectorizer(min_df=minValue, stop_words='english', max_df=maxValue, max_features=totalFeatures)
    tfIDF = vector.fit_transform(textData)
    lda_allocation = LatentDirichletAllocation(max_iter=iteration,n_components=num_comp, learning_offset=offset, learning_method=method, random_state=state)
    lda_allocation.fit(tfIDF)
    features = vector.get_feature_names()
    output = []
    target = np.zeros(20)
    predict = np.zeros(20)
    for index, name in enumerate(lda_allocation.components_):
        topic="Topic No %d: " % index
        topic+=" ".join([features[j] for j in name.argsort()[:-number_of_Words - 1:-1]])
        text.insert(END,topic+"\n")
        output.append(" ".join([features[j] for j in name.argsort()[:-number_of_Words - 1:-1]]))
    count = 0
    for i in range(len(output)):
        arr = output[i].split(" ")
        flag = False
        for j in range(len(arr)):
            if arr[j] in labels:
                count += 1
                flag = True
        if flag == False:
            predict[i] = 1
                
    p = precision_score(target, predict,average='macro')
    r = recall_score(target, predict,average='macro')
    f = f1_score(target, predict,average='macro') 
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    pmi_score.append(count/len(output))
    tc_score.append(accuracy_score(predict,target))
    text.insert(END,"LDA Precision: "+str(p)+"\n")
    text.insert(END,"LDA Recall   : "+str(r)+"\n")
    text.insert(END,"LDA FScore   : "+str(f)+"\n")
    text.insert(END,"LDA PMI      : "+str(count/len(output))+"\n")
    text.insert(END,"LDA TC       : "+str(accuracy_score(predict,target))+"\n\n")


def graph():
    text.delete('1.0', END)
    text.insert(END,"ClusTop-Word-NA CO-WORD Precision: "+str(precision[0])+"\n")
    text.insert(END,"ClusTop-Word-NA CO-WORD Recall   : "+str(recall[0])+"\n")
    text.insert(END,"ClusTop-Word-NA CO-WORD FScore   : "+str(fscore[0])+"\n")
    text.insert(END,"ClusTop-Word-NA CO-WORD PMI      : "+str(pmi_score[0])+"\n")
    text.insert(END,"ClusTop-Word-NA CO-WORD TC       : "+str(tc_score[0])+"\n\n")

    text.insert(END,"ClusTop-Word-NA Bigram Precision: "+str(precision[1])+"\n")
    text.insert(END,"ClusTop-Word-NA Bigram Recall   : "+str(recall[1])+"\n")
    text.insert(END,"ClusTop-Word-NA Bigram FScore   : "+str(fscore[1])+"\n")
    text.insert(END,"ClusTop-Word-NA Bigram PMI      : "+str(pmi_score[1])+"\n")
    text.insert(END,"ClusTop-Word-NA Bigram TC       : "+str(tc_score[1])+"\n\n")

    text.insert(END,"LDA Precision: "+str(precision[2])+"\n")
    text.insert(END,"LDA Recall   : "+str(recall[2])+"\n")
    text.insert(END,"LDA FScore   : "+str(fscore[2])+"\n")
    text.insert(END,"LDA PMI      : "+str(pmi_score[2])+"\n")
    text.insert(END,"LDA TC       : "+str(tc_score[2])+"\n\n")
    
    df = pd.DataFrame([['Coword','Precision',precision[0]],['Coword','Recall',recall[0]],['Coword','F1 Score',fscore[0]],['Coword','PMI',pmi_score[0]],['Coword','TC',tc_score[0]],
                       ['Bigram','Precision',precision[1]],['Bigram','Recall',recall[1]],['Bigram','F1 Score',fscore[1]],['Bigram','PMI',pmi_score[1]],['Bigram','TC',tc_score[1]],
                       ['LDA','Precision',precision[2]],['LDA','Recall',recall[2]],['LDA','F1 Score',fscore[2]],['LDA','PMI',pmi_score[2]],['LDA','TC',tc_score[2]],
                       
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()

def table():
    output = "<html><body><center><br/><table border=1><tr><th>Algorithm Name</th><th>Precision %</th><th>Recall %</th><th>F1Score %</th><th>PMI %</th><th>TC %</th>"
    output+="</tr>"
    output+='<tr><td>ClusTop-Word-NA Coword</td><td>'+str(precision[0])+'</td><td>'+str(recall[0])+'</td><td>'+str(fscore[0])+'</td><td>'+str(pmi_score[0])+'</td><td>'+str(tc_score[0])+'</td></tr>'
    output+='<tr><td>ClusTop-BiG-NA Bigram</td><td>'+str(precision[1])+'</td><td>'+str(recall[1])+'</td><td>'+str(fscore[1])+'</td><td>'+str(pmi_score[1])+'</td><td>'+str(tc_score[1])+'</td></tr>'
    output+='<tr><td>LDA</td><td>'+str(precision[2])+'</td><td>'+str(recall[2])+'</td><td>'+str(fscore[2])+'</td><td>'+str(pmi_score[2])+'</td><td>'+str(tc_score[2])+'</td></tr>'
    output+='</table></body></html>'

    f = open("table.html", "w")
    f.write(output)
    f.close()
    webbrowser.open("table.html",new=2)

def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='ClusTop: A Clustering-based Topic Modelling Algorithm for Twitter using Word Networks',anchor=W, justify=LEFT)
title.config(bg='black', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 13, 'bold')

loadwordnetButton = Button(main, text="Upload Tweet Dataset", command=uploadDataset)
loadwordnetButton.place(x=50,y=100)
loadwordnetButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=350)


uploadButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
uploadButton.place(x=50,y=150)
uploadButton.config(font=font1)

cowordButton = Button(main, text="Run ClusTop-Word-NA Algorithm", command=cowordClusTop)
cowordButton.place(x=350,y=150)
cowordButton.config(font=font1)

BIButton = Button(main, text="Run ClusTop-BiG-NA Algorithm", command=biwordClusTop)
BIButton.place(x=720,y=150)
BIButton.config(font=font1)

ldaButton = Button(main, text="Run LDA Algorithm", command=runLDA)
ldaButton.place(x=50,y=200)
ldaButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=350,y=200)
graphButton.config(font=font1)


tableButton = Button(main, text="Comparison Table", command=table)
tableButton.place(x=720,y=200)
tableButton.config(font=font1)

closeButton = Button(main, text="Exit", command=close)
closeButton.place(x=50,y=250)
closeButton.config(font=font1)
                    

text=Text(main,height=15,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1)

main.config(bg='chocolate1')
main.mainloop()
