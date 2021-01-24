import json
import spacy
from collections import Counter
import numpy as np
nlp = spacy.load("en_core_web_sm")

def find_index(sent, word, length):
    start= []
    sent, word = [[t.lower() for t in tokens] for tokens in (sent, word)]
    for i in range(len(sent)):
        get_index = []
        i_index = 0
        for j in range(length):
            if i + j < len(sent):
                if sent[i + j]==word[j]:
                    get_index.append(word[j])
                    i_index = i
            elif sent[i]!=word[j]:
                break
        if len(get_index) == length:
            start.append(i_index)
    end=[i+length-1 for i in start]
    offset=[list(i) for i in zip(start,end)]

    return offset

word_pairs = {"it's":"it is", "don't":"do not", "doesn't":"does not", "didn't":"did not", "you'd":"you would",\
              "you're":"you are", "you'll":"you will", "i'm":"i am", "they're":"they are", "that's":"that is", \
              "what's":"what is", "couldn't":"could not", "i've":"i have", "we've":"we have", "can't":"can not", \
              "i'd":"i would", "aren't":"are not", "isn't":"is not", "wasn't":"was not", "that's":"that is",\
              "weren't":"were not", "won't":"will not", "there's":"there is", "there're":"there are","let's":"let us",\
              "wouldn't":"would not", "where's":" where is", "we're":"we are", "let's":"let us","you've":"you have",\
              "shouldn't":"should not","hadn't":"had not","haven't":"have not","y'know":'you know',"goin'":"going",\
              "i'll":'i will'}

def find_offset_index():
    print("finding_extra_data")
    pedc=json.load(open('extra_data.json','r',encoding='utf8'))
    #print(len(pedc))
    for d in pedc:
        for dd in d[1]:
            sent_id=dd['sent_id']
            if sent_id>= len(d[0]):
                print(d[0])
            utter=d[0][sent_id].lower()
            for k, v in word_pairs.items():
                utter = utter.replace(k, v)
            break_index = utter.find(':')
            speaker = utter[:break_index]
            utters=nlp(utter)

            trigger=[str(token) for token in nlp(dd['trigger_word'])]
            length=len(trigger)
            tokens=[str(token) for token in utters]
            offset=find_index(tokens,trigger,length)
            if offset ==[]:
                print(utter)
                print(trigger)
                continue
            dd['offset']=offset[0]
            dd["type"]=[]
            dd['speaker_name']=speaker

    with open("extra_datas.json",'w',encoding='utf8')as of:
        json.dump(pedc,of, indent=1, ensure_ascii=False)

find_offset_index()



def speaker_name():
    pedc=json.load(open('PEDC_formlike_DialogRE_spacy.json','r',encoding='utf8'))
    #print(len(pedc))
    for d in pedc:
        for dd in d[1]:
            sent_id=dd['sent_id']
            utter=d[0][sent_id].lower()
            for k, v in word_pairs.items():
                utter = utter.replace(k, v)
            break_index = utter.find(':')
            speaker, utter = utter[:break_index], utter[break_index:]
            dd['speaker_name']=speaker
            len1=len(speaker.split())
            speaker = ''.join(speaker.split())  # remove white space
            len2=len(speaker.split())
            distance=len1-len2
            dd['offset_speakername']=[i-distance for i in dd['offset']]
            if dd['type']==[]:
                print(dd)
#speaker_name()

def cal_type():
    types=[]
    pedc = json.load(open('PEDC_formlike_DialogRE_spacy_name2id.json', 'r', encoding='utf8'))
    print(len(pedc))
    for d in pedc:
        for dd in d[1]:

            types+=dd['type']

    type_counter=Counter(types)
    print(type_counter)
    with open("types.json",'w',encoding='utf8')as of:
        json.dump(type_counter,of, indent=1, ensure_ascii=False)

#cal_type()

# def cal_type111():
#     types=[]
#     pedc = json.load(open('PEDC_formlike_DialogRE_spacy_name2id.json', 'r', encoding='utf8'))
#     print(len(pedc))
#     for d in pedc:
#         for idx, dd in enumerate(d[1]):
#             for word in ['know','knew','knows','known','knowing']:
#                 if dd['trigger_word']==word:
#                     del d[1][idx]
#                     continue
#
#     with open("pedc.json",'w',encoding='utf8')as of:
#         json.dump(pedc,of, indent=1, ensure_ascii=False)
#
# #cal_type111()
