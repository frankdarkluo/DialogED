Training with the scrips:

run BiLSTM 
```
python run_classifier.py --lstm_only --bilstm_crf False
```

run BilSTM+CRF
```
python run_classifier.py --lstm_only --bilstm_crf True
```

# DialogED
# 1.在DialogRE目录下新建一个文件夹名为 bert_base，并将bert pretrained的多个文件解压到此处 
# 2.在DialogRE/GDPNet/data中需要下载 glove.6B.300d.txt


