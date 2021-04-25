# 
# CUDA_VISIBLE_DEVICES=5 python ./evaluate_validation_curve.py --model="gpt2" --model_checkpoint ./runs/Jul09_14-54-22_language_gpt2 --max_history=1 --dataset_path data/json_transfertransfo_crisisbot.json
# CUDA_VISIBLE_DEVICES=5 python ./evaluate_validation_curve.py --model="gpt2" --model_checkpoint ./runs/Jul09_14-56-03_language_gpt2 --max_history=1 --dataset_path data/json_transfertransfo_crisisbot.json
# CUDA_VISIBLE_DEVICES=5 python ./evaluate_validation_curve.py --model="gpt2" --model_checkpoint ./runs/Jul09_14-57-13_language_gpt2 --max_history=1 --dataset_path data/json_transfertransfo_crisisbot.json
# CUDA_VISIBLE_DEVICES=5 python ./evaluate_validation_curve.py --model="gpt2" --model_checkpoint ./runs/Jul09_14-58-26_language_gpt2 --max_history=1 --dataset_path data/json_transfertransfo_crisisbot.json


# python ./evaluate_validation_curve.py --model="gpt2" --model_checkpoint ./runs/Jul10_22-25-30_language_gpt2 --max_history=1 --dataset_path data/json_transfertransfo_crisisbot_v1.json
# python ./evaluate_validation_curve.py --model="gpt2" --model_checkpoint ./runs/Jul11_05-07-53_language_gpt2 --max_history=1 --dataset_path data/json_transfertransfo_crisisbot_v1.json
# python ./evaluate_validation_curve.py --model="gpt2" --model_checkpoint ./runs/Jul10_21-38-06_language_gpt2 --max_history=1 --dataset_path data/json_transfertransfo_crisisbot_v1.json
# python ./evaluate_validation_curve.py --model="gpt2" --model_checkpoint ./runs/Jul11_00-24-13_language_gpt2 --max_history=1 --dataset_path data/json_transfertransfo_crisisbot_v1.json
# python ./evaluate_validation_curve.py --model="gpt2" --model_checkpoint ./runs/Jul11_00-40-24_language_gpt2 --max_history=1 --dataset_path data/json_transfertransfo_crisisbot_v1.json



# python ./train_distil.py --model_class="gpt2" --model_checkpoint="gpt2" --train_batch_size=1 --valid_batch_size=1 --gradient_accumulation_steps=4 --lm_coef=2.0 --mc_coef=1.0 --max_history=1 --n_epochs=7 --num_candidates=1 --personality_permutations=1 --dataset_path data/json_transfertransfo_crisisbot_v1.json --h_coef=0.1




CUDA_VISIBLE_DEVICES=7 python ./train_distil.py --model_class="gpt2" --model_checkpoint="gpt2" --train_batch_size=1 --valid_batch_size=1 --gradient_accumulation_steps=4 --lm_coef=2.0 --mc_coef=0.0 --max_history=1 --n_epochs=7 --num_candidates=1 --personality_permutations=1 --dataset_path data/json_transfertransfo_crisisbot_v1.json --h_coef=1.0
# python ./train_distil.py --model_class="gpt2" --model_checkpoint="gpt2" --train_batch_size=1 --valid_batch_size=1 --gradient_accumulation_steps=4 --lm_coef=2.0 --mc_coef=0.0 --max_history=1 --n_epochs=7 --num_candidates=1 --personality_permutations=1 --dataset_path data/json_transfertransfo_crisisbot_v1.json --h_coef=0.5
# python ./train_distil.py --model_class="gpt2" --model_checkpoint="gpt2" --train_batch_size=1 --valid_batch_size=1 --gradient_accumulation_steps=4 --lm_coef=2.0 --mc_coef=0.0 --max_history=1 --n_epochs=7 --num_candidates=1 --personality_permutations=1 --dataset_path data/json_transfertransfo_crisisbot_v1.json --h_coef=1.0

# python ./train_distil.py --model_class="gpt2" --model_checkpoint="gpt2" --train_batch_size=1 --valid_batch_size=1 --gradient_accumulation_steps=4 --lm_coef=2.0 --mc_coef=0.0 --h_coef=0.0 --max_history=1 --n_epochs=7 --num_candidates=1 --personality_permutations=1 --dataset_path data/json_transfertransfo_crisisbot_v1.json --num_candidates_valid=10
# python ./train_distil.py --model_class="gpt2" --model_checkpoint="gpt2" --train_batch_size=1 --valid_batch_size=1 --gradient_accumulation_steps=4 --lm_coef=2.0 --mc_coef=1.0 --h_coef=0.0 --max_history=1 --n_epochs=7 --num_candidates=4 --personality_permutations=1 --dataset_path data/json_transfertransfo_crisisbot_v1.json --num_candidates_valid=10


# python ./evaluate_validation_curve.py --model="gpt2" --model_checkpoint ./runs/Jul13_07-10-13_language_gpt2 --max_history=1 --dataset_path data/json_transfertransfo_crisisbot_v1.json

