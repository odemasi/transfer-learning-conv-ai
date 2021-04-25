# CUDA_VISIBLE_DEVICES=7 ./tune_distilgpt2_h_penalty.sh

python ./train_distil.py --model_class="distilgpt2" --model_checkpoint="distilgpt2" --train_batch_size=2 --valid_batch_size=2 --gradient_accumulation_steps=4 --lm_coef=2.0 --mc_coef=0.0 --h_coef=2.0 --max_history=2 --n_epochs=7 --num_candidates=1 --personality_permutations=2 