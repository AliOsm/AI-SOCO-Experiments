curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
python pytorch-xla-env-setup.py --version "nightly"

pip install git+https://github.com/huggingface/transformers.git
git clone https://github.com/huggingface/transformers.git

python transformers/examples/xla_spawn.py --num_cores 8 \
	transformers/examples/language-modeling/run_language_modeling.py \
    --output_dir=AI-SOCO-Experiments/models_dir/roberta-small-code-v1 \
    --model_type=roberta \
    --do_train \
    --train_data_file=AI-SOCO-Experiments/data_dir/train.cat \
    --do_eval \
    --eval_data_file=AI-SOCO-Experiments/data_dir/dev.cat \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --config_name=AI-SOCO-Experiments/models_dir/roberta-small-code \
    --tokenizer_name=AI-SOCO-Experiments/models_dir/roberta-small-code \
    --evaluate_during_training \
    --num_train_epochs=10 \
    --warmup_steps=500 \
    --save_steps=2500 \
    --save_total_limit=3 \
    --eval_steps=2500
