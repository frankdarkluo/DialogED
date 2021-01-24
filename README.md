Training with the scrips:

python run_classifier.py

# GDPNet-Improved
# 1.在DialogRE目录下新建一个文件夹名为 bert_base，并将bert pretrained的多个文件解压到此处 
# 2.在DialogRE/GDPNet/data中需要下载 glove.6B.300d.txt
python run_classifier.py   --task_name lstm  --do_train --do_eval   --data_dir .   --vocab_file ../bert_base/vocab.txt   --bert_config_file ../bert_base/bert_config.json   --init_checkpoint ../bert_base/pytorch_model.bin   --max_seq_length 512   --train_batch_size 24   --learning_rate 1e-4   --num_train_epochs 40.0   --output_dir lstm_f1  --gradient_accumulation_steps 1 --encoder_type LSTM
