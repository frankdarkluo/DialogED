export BERT_BASE_DIR=../bert_base

python convert_tf_checkpoint_to_pytorch.py --tf_checkpoint_path=$BERT_BASE_DIR/bert_model.ckpt --bert_config_file=$BERT_BASE_DIR/bert_config.json --pytorch_dump_path=$BERT_BASE_DIR/pytorch_model.bin

python run_classifier.py   --task_name berts  --do_train --do_eval   --data_dir .   --vocab_file $BERT_BASE_DIR/vocab.txt   --bert_config_file $BERT_BASE_DIR/bert_config.json   --init_checkpoint $BERT_BASE_DIR/pytorch_model.bin   --max_seq_length 512   --train_batch_size 24   --learning_rate 3e-5   --num_train_epochs 20.0   --output_dir berts_f1  --gradient_accumulation_steps 6

rm berts_f1/model_best.pt
cp -r berts_f1 berts_f1c
python run_classifier.py   --task_name bertsf1c --do_eval   --data_dir .   --vocab_file $BERT_BASE_DIR/vocab.txt   --bert_config_file $BERT_BASE_DIR/bert_config.json   --init_checkpoint $BERT_BASE_DIR/pytorch_model.bin   --max_seq_length 512   --train_batch_size 24   --learning_rate 3e-5   --num_train_epochs 20.0   --output_dir berts_f1c  --gradient_accumulation_steps 6
python evaluate.py --f1dev berts_f1/logits_dev.txt --f1test berts_f1/logits_test.txt --f1cdev berts_f1c/logits_dev.txt --f1ctest berts_f1c/logits_test.txt





CUDA_VISIBLE_DIVICES=3 python run_classifier.py   --task_name lstm  --do_train --do_eval   --data_dir .   --vocab_file ../bert_base/vocab.txt   --bert_config_file ../bert_base/bert_config.json   --init_checkpoint ../bert_base/pytorch_model.bin   --max_seq_length 512   --train_batch_size 24   --learning_rate 1e-4   --num_train_epochs 40.0   --output_dir lstm_f1  --gradient_accumulation_steps 1 --encoder_type LSTM

rm lstm_f1/model_best.pt
cp -r lstm_f1 lstm_f1c

CUDA_VISIBLE_DIVICES=3 python run_classifier.py   --task_name lstmf1c   --do_eval   --data_dir .   --vocab_file ../bert_base/vocab.txt   --bert_config_file ../bert_base/bert_config.json   --init_checkpoint ../bert_base/pytorch_model.bin   --max_seq_length 512   --train_batch_size 24   --learning_rate 1e-4   --num_train_epochs 20.0   --output_dir lstm_f1c  --gradient_accumulation_steps 1 --encoder_type LSTM

python evaluate.py --f1dev lstm_f1/logits_dev.txt --f1test lstm_f1/logits_test.txt --f1cdev lstm_f1c/logits_dev.txt --f1ctest lstm_f1c/logits_test.txt

------------------------------------------

CUDA_VISIBLE_DIVICES=1 python run_classifier.py   --task_name roberta  --do_train --do_eval   --data_dir .   --vocab_file ../bert_base/vocab.txt   --bert_config_file ../bert_base/bert_config.json   --init_checkpoint ../bert_base/pytorch_model.bin   --max_seq_length 512   --train_batch_size 16   --learning_rate 3e-5   --num_train_epochs 20.0   --output_dir roberta_f1  --gradient_accumulation_steps 1 --encoder_type Bert

rm roberta_f1/model_best.pt
cp -r roberta_f1 roberta_f1c

CUDA_VISIBLE_DIVICES=3 python run_classifier.py   --task_name robertaf1c   --do_eval   --data_dir .   --vocab_file ../bert_base/vocab.txt   --bert_config_file ../bert_base/bert_config.json   --init_checkpoint ../bert_base/pytorch_model.bin   --max_seq_length 512   --train_batch_size 24   --learning_rate 3e-5   --num_train_epochs 20.0   --output_dir roberta_f1c  --gradient_accumulation_steps 1 --encoder_type Bert

python evaluate.py --f1dev roberta_f1/logits_dev.txt --f1test roberta_f1/logits_test.txt --f1cdev roberta_f1c/logits_dev.txt --f1ctest roberta_f1c/logits_test.txt

-------------------------------------------
use bert tokenizer but load from pretrained roberta

python run_classifier.py   --task_name bert  --do_train --do_eval   --data_dir .   --vocab_file ../roberta_base/vocab.txt   --bert_config_file ../roberta_base/config.json   --init_checkpoint ../roberta_base/pytorch_model.bin   --max_seq_length 512   --train_batch_size 16   --learning_rate 3e-5   --num_train_epochs 20.0   --output_dir pretrained_roberta_f1  --gradient_accumulation_steps 1 --encoder_type Bert

rm pretrained_roberta_f1/model_best.pt
cp -r pretrained_roberta_f1 pretrained_roberta_f1c

python run_classifier.py   --task_name bertf1c   --do_eval   --data_dir .   --vocab_file ../roberta_base/vocab.txt   --bert_config_file ../roberta_base/config.json   --init_checkpoint ../roberta_base/pytorch_model.bin   --max_seq_length 512   --train_batch_size 24   --learning_rate 3e-5   --num_train_epochs 20.0   --output_dir pretrained_roberta_f1c  --gradient_accumulation_steps 1 --encoder_type Bert

-------------------------------------------

python run_classifier.py   --task_name lstm  --do_train --do_eval   --data_dir .   --vocab_file ../bert_base/vocab.txt   --bert_config_file ../bert_base/bert_config.json   --init_checkpoint ../bert_base/pytorch_model.bin   --max_seq_length 512   --train_batch_size 20   --learning_rate 1e-4    --num_train_epochs 30.0   --output_dir lstm_only_f1  --gradient_accumulation_steps 1 --encoder_type LSTM --lstm_only