python train_mlm.py \
    --model_name_or_path roberta-base \
    --train_file data/corpus_train.txt \
    --validation_file data/corpus_dev.txt \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-mlm