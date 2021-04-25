# CUDA_VISIBLE_DEVICES=7 

# python ./train_distil.py --model_class="gpt2" --model_checkpoint="gpt2" --train_batch_size=2 --valid_batch_size=2 --gradient_accumulation_steps=4 --lm_coef=2.0 --mc_coef=1.0 --max_history=2 --n_epochs=7 --num_candidates=4 --personality_permutations=2 



# CUDA_VISIBLE_DEVICES=2 python ./train_distil.py --model_class="gpt2" --model_checkpoint="gpt2" --train_batch_size=1 --valid_batch_size=1 --gradient_accumulation_steps=4 --lm_coef=2.0 --mc_coef=0.0 --h_coef=0.0 --max_history=1 --n_epochs=7 --num_candidates=4 --personality_permutations=1 --dataset_path data/json_transfertransfo_crisisbot.json --tune_head_only
# 
# 
# CUDA_VISIBLE_DEVICES=2 python ./train_distil.py --model_class="gpt2" --model_checkpoint="gpt2" --train_batch_size=1 --valid_batch_size=1 --gradient_accumulation_steps=4 --lm_coef=2.0 --mc_coef=1.0 --h_coef=0.0 --max_history=1 --n_epochs=7 --num_candidates=4 --personality_permutations=1 --dataset_path data/json_transfertransfo_crisisbot.json --tune_head_only
# 
# 
# CUDA_VISIBLE_DEVICES=2 python ./train_distil.py --model_class="gpt2" --model_checkpoint="gpt2" --train_batch_size=1 --valid_batch_size=1 --gradient_accumulation_steps=4 --lm_coef=2.0 --mc_coef=0.0 --max_history=1 --n_epochs=7 --num_candidates=1 --personality_permutations=1 --dataset_path data/json_transfertransfo_crisisbot.json --h_coef=0.1  --tune_head_only



CUDA_VISIBLE_DEVICES=2 python ./train_distil.py --model_class="gpt2" --model_checkpoint="gpt2" --train_batch_size=2 --valid_batch_size=1 --gradient_accumulation_steps=4 --lm_coef=1.0 --truncate_input=512 --n_epochs=10 --num_candidates=1 --personality_permutations=1 --mc_coef=0.0 --h_coef=0.0 --dataset_path data/json_transfertransfo_crisisbot_v1.json
