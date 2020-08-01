!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version "nightly"

!pip install git+https://github.com/huggingface/transformers.git
!git clone https://github.com/huggingface/transformers.git

!git clone https://github.com/AliOsm/AI-SOCO-Experiments.git

!python transformers/examples/xla_spawn.py --num_cores 8 \
	transformers/examples/language-modeling/run_language_modeling.py \
    --output_dir=output \
    --model_type=roberta \
    --do_train \
    --train_data_file=AI-SOCO-Experiments/data_dir/train.cat \
    --do_eval \
    --eval_data_file=AI-SOCO-Experiments/data_dir/dev.cat \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --tokenizer_name=AI-SOCO-Experiments/tokenizer \
    --evaluate_during_training=True \
    --num_train_epochs=10 \
    --warmup_steps=500 \
    --save_steps=2500 \
    --save_total_limit=3 \
    --eval_steps=2500
