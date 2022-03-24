## Train NER Models

This code is based on the official code ([link](https://github.com/dmis-lab/biobert-pytorch/tree/master/named-entity-recognition)) for the PyTorch version of BioBERT.

### Requirements
* transformers==3.0.0

### Training
```bash
export SAVE_DIR=./output
export DATA_DIR=../datasets/NER

export MAX_LENGTH=192
export BATCH_SIZE=32
export NUM_EPOCHS=30
export SAVE_STEPS=1000
export ENTITY=NCBI-disease
export SEED=1

python run_ner.py \
    --data_dir ${DATA_DIR}/${ENTITY}/ \
    --labels ${DATA_DIR}/${ENTITY}/labels.txt \
    --model_name_or_path dmis-lab/biobert-base-cased-v1.1 \
    --output_dir ${SAVE_DIR}/${ENTITY} \
    --max_seq_length ${MAX_LENGTH} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --save_steps ${SAVE_STEPS} \
    --seed ${SEED} \
    --do_train \
    --do_eval \
    --do_predict \
    --overwrite_output_dir
```

### Debiasing

To be updated soon.
