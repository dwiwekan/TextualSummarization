from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import re
from string import punctuation
from keras_preprocessing.sequence import pad_sequences
import pickle
from keras.models import load_model
from nltk import tokenize
import pandas as pd
import numpy as np
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class TextualSummarization(object):
    def __init__(self):
        self.model = load_model("SNLI_model_final.h5")
        self.model.load_weights('SNLI_weight_final.hdf5')
        self.tokenizer = pickle.load(open('tokenizer.pickle', 'rb'))
    
    def clean_text(self, text):
        text = text.lower().split() # lowercase
        text = " ".join(text)
        #remove punct
        text = re.sub("[^A-Za-z']+", ' ', str(text)).replace("'", '')
        text = re.sub(r"\bum*\b", "", text)
        text = re.sub(r"\buh*\b", "", text)
        text = re.sub(r"won\'t", "will not", text)
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
        
        text = text.translate(str.maketrans('', '', punctuation))
        return text.strip()
    
    def PadSeq(self, text):
        SentenceLen = 100
        sequences = self.tokenizer.texts_to_sequences(text)
        return pad_sequences(sequences, maxlen=SentenceLen, padding='post', truncating='post')

    def predict(self, text, persentase, awal_akhir, posisi):
        split = tokenize.sent_tokenize(text)
        # Pemilihan empat skenario
        list_kalimat = []
        if awal_akhir == 1 and posisi == 1:
            list_kalimat = [[split[0], i] for i in split[1:]]
        elif awal_akhir == 1 and posisi == 2:
            list_kalimat = [[i, split[0]] for i in split[1:]]
        elif awal_akhir == 2 and posisi == 1:
            list_kalimat = [[split[-1], i] for i in split[:-1]]
        else:
            list_kalimat = [[i, split[-1]] for i in split[:-1]]
        df_kalimat = pd.DataFrame(list_kalimat,columns=['premis','hipotesis'])
        if awal_akhir == 1 or awal_akhir == 2:
            df_kalimat = df_kalimat.iloc[1:]
        else:
            df_kalimat = df_kalimat.iloc[:-1]
        df_kalimat.reset_index(drop=True,inplace=True)

        df_clean = df_kalimat.copy()
        df_clean['premis'] = df_kalimat['premis'].astype(str).apply(lambda text: self.clean_text(text))
        df_clean['hipotesis'] = df_kalimat['hipotesis'].astype(str).apply(lambda text: self.clean_text(text))
        # insert data to tuple
        df_clean = df_clean['premis'].tolist(),df_clean['hipotesis'].tolist()
        # df_clean[1]
        test_x = self.PadSeq(df_clean[0]), self.PadSeq(df_clean[1])
        # Get result on probability
        test_pred = self.model.predict(test_x)
        format_string = "{:.2f}"
        formatter = np.vectorize(format_string.format)
        predictions_formatted = formatter(test_pred*100)
        entailment_percentage = list()
        for i in predictions_formatted:
            entailment_percentage.append(float(i[0]))
        # Create dataframe
        df_kata = pd.DataFrame({'premis':df_kalimat['premis'],'hipotesis':df_kalimat['hipotesis']})
        df_kata = df_kata.reset_index(drop=True)
        df_pred = pd.DataFrame({'entailment':entailment_percentage})
        df_full = pd.concat([df_kata,df_pred],axis=1)
        prediksi_list = list(enumerate(df_full['entailment']))
        prediksi_list.sort(key=lambda x: x[1], reverse=True) 
        # Persentase
        jumlah_kalimat = round(len(prediksi_list)*persentase)

        kalimat_ringkasan = prediksi_list[:jumlah_kalimat]
        kalimat_ringkasan.sort(key=lambda x: x[0], reverse=False)   

        kalimat_akhir = list()
        for idx,row in df_full.iterrows():
            if idx in [i[0] for i in kalimat_ringkasan]:
                if posisi == 1:
                    kalimat_akhir.append(row['hipotesis'])
                else:
                    kalimat_akhir.append(row['premis'])

        if posisi == 1:
            summary = (f"{df_full['premis'][0]} ")
        else:
            summary = (f"{df_full['hipotesis'][0]} ")

        for idx,item in enumerate(kalimat_akhir):
            if idx==0:
                summary = summary + item
            else:
                summary = summary + ' '+item
        list_df = df_full.values.tolist()
        # list_all = ([summary, list_df])
        return {"summary": summary, "list_df": list_df}


# text = "Quarterly profits at US media giant TimeWarner jumped 76% to $1.13bn (£600m) for the three months to December, from $639m year-earlier. TimeWarner said fourth quarter sales rose 2% to $11.1bn from $10.9bn. Its profits were buoyed by one-off gains which offset a profit dip at Warner Bros, and less users for AOL. Time Warner said on Friday that it now owns 8% of search-engine Google. But its own internet business, AOL, had has mixed fortunes. It lost 464,000 subscribers in the fourth quarter profits were lower than in the preceding three quarters. However, the company said AOL's underlying profit before exceptional items rose 8% on the back of stronger internet advertising revenues. It hopes to increase subscribers by offering the online service free to TimeWarner internet customers and will try to sign up AOL's existing customers for high-speed broadband. TimeWarner also has to restate 2000 and 2003 results following a probe by the US Securities Exchange Commission (SEC), which is close to concluding."
# persentase = 0.3
# awal_akhir = 1
# posisi = 1

# TextPredict = TextualSummarization()
# summary = TextPredict.predict(text, persentase, awal_akhir, posisi)