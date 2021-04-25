# python ./train_distil.py --model_class="distilgpt2" --model_checkpoint="distilgpt2" --train_batch_size=2 --valid_batch_size=2 --gradient_accumulation_steps=4 --lm_coef=2.0 --mc_coef=1.0 --max_history=2 --n_epochs=7 --num_candidates=4 --personality_permutations=2 


CUDA_VISIBLE_DEVICES=3 python ./train_distil.py --model_class="distilgpt2" --model_checkpoint="distilgpt2" --train_batch_size=1 --valid_batch_size=1 --gradient_accumulation_steps=4 --lm_coef=2.0 --mc_coef=0.0 --max_history=1 --n_epochs=7 --num_candidates=1 --personality_permutations=1 --dataset_path data/json_transfertransfo_crisisbot.json --h_coef=0.01 


CUDA_VISIBLE_DEVICES=3 python ./train_distil.py --model_class="distilgpt2" --model_checkpoint="distilgpt2" --train_batch_size=1 --valid_batch_size=1 --gradient_accumulation_steps=4 --lm_coef=2.0 --mc_coef=0.0 --max_history=1 --n_epochs=7 --num_candidates=1 --personality_permutations=1 --dataset_path data/json_transfertransfo_crisisbot.json --h_coef=0.1


CUDA_VISIBLE_DEVICES=3 python ./train_distil.py --model_class="distilgpt2" --model_checkpoint="distilgpt2" --train_batch_size=1 --valid_batch_size=1 --gradient_accumulation_steps=4 --lm_coef=2.0 --mc_coef=0.0 --max_history=1 --n_epochs=7 --num_candidates=1 --personality_permutations=1 --dataset_path data/json_transfertransfo_crisisbot.json --h_coef=0.5


CUDA_VISIBLE_DEVICES=3 python ./train_distil.py --model_class="distilgpt2" --model_checkpoint="distilgpt2" --train_batch_size=1 --valid_batch_size=1 --gradient_accumulation_steps=4 --lm_coef=2.0 --mc_coef=0.0 --max_history=1 --n_epochs=7 --num_candidates=1 --personality_permutations=1 --dataset_path data/json_transfertransfo_crisisbot.json --h_coef=1.0