module --force purge
module load arcc/1.0
module load gcc/14.2.0
module load miniconda3
module load cuda-toolkit/12.8.0

source activate /gscratch/tkishore/vllm 



export XDG_CACHE_HOME=/gscratch/tkishore/cache
export HF_HOME=/gscratch/tkishore/cache/hf_cache
export HF_HUB_CACHE=/gscratch/tkishore/cache/hf_cache/hub
export TORCH_HOME=/gscratch/tkishore/cache/torch
export FLASHINFER_CACHE_DIR=/gscratch/tkishore/cache/flashinfer


#Qwen3-VL-30B-A3B-Instruct
python eval.py --cal_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_Qwen_Qwen3-VL-30B-A3B-Instruct/Figures/UQ/20260320_054415/score/cal_predictions.json \
               --test_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_Qwen_Qwen3-VL-30B-A3B-Instruct/Figures/UQ/20260320_054415/score/test_predictions.json \
               --out_dir /gscratch/tkishore/MLLM-Judge/Experiments/eval/Qwen_Qwen3-VL-30B-A3B-Instruct/   --top_k_score 5   --top_k_pair 3  --task_type score --model_name Qwen3-VL-30B-A3B-Instruct

python eval.py --cal_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_Qwen_Qwen3-VL-30B-A3B-Instruct/Figures/UQ/20260320_054602/pair/cal_predictions.json \
               --test_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_Qwen_Qwen3-VL-30B-A3B-Instruct/Figures/UQ/20260320_054602/pair/test_predictions.json \
               --out_dir /gscratch/tkishore/MLLM-Judge/Experiments/eval/Qwen_Qwen3-VL-30B-A3B-Instruct/   --top_k_score 5   --top_k_pair 3  --task_type pair --model_name Qwen3-VL-30B-A3B-Instruct

python eval.py --cal_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_Qwen_Qwen3-VL-30B-A3B-Instruct/Figures/UQ/20260320_054746/batch/cal_predictions.json \
               --test_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_Qwen_Qwen3-VL-30B-A3B-Instruct/Figures/UQ/20260320_054746/batch/test_predictions.json \
               --out_dir /gscratch/tkishore/MLLM-Judge/Experiments/eval/Qwen_Qwen3-VL-30B-A3B-Instruct/   --top_k_score 5   --top_k_pair 3  --task_type batch --model_name Qwen3-VL-30B-A3B-Instruct


#Qwen3-VL-8B-Instruct
python eval.py --cal_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_Qwen_Qwen3-VL-8B-Instruct/Figures/UQ/20260320_052807/score/cal_predictions.json \
               --test_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_Qwen_Qwen3-VL-8B-Instruct/Figures/UQ/20260320_052807/score/test_predictions.json \
               --out_dir /gscratch/tkishore/MLLM-Judge/Experiments/eval/Qwen_Qwen3-VL-8B-Instruct/   --top_k_score 5   --top_k_pair 3  --task_type score --model_name Qwen3-VL-8B-Instruct

python eval.py --cal_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_Qwen_Qwen3-VL-8B-Instruct/Figures/UQ/20260320_052955/pair/cal_predictions.json \
               --test_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_Qwen_Qwen3-VL-8B-Instruct/Figures/UQ/20260320_052955/pair/test_predictions.json \
               --out_dir /gscratch/tkishore/MLLM-Judge/Experiments/eval/Qwen_Qwen3-VL-8B-Instruct/   --top_k_score 5   --top_k_pair 3  --task_type pair --model_name Qwen3-VL-8B-Instruct

python eval.py --cal_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_Qwen_Qwen3-VL-8B-Instruct/Figures/UQ/20260320_053140/batch/cal_predictions.json \
               --test_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_Qwen_Qwen3-VL-8B-Instruct/Figures/UQ/20260320_053140/batch/test_predictions.json \
               --out_dir /gscratch/tkishore/MLLM-Judge/Experiments/eval/Qwen_Qwen3-VL-8B-Instruct/   --top_k_score 5   --top_k_pair 3  --task_type batch --model_name Qwen3-VL-8B-Instruct


#Qwen3-VL-4B-Instruct
python eval.py --cal_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_Qwen_Qwen3-VL-4B-Instruct/Figures/UQ/20260320_051336/score/cal_predictions.json \
               --test_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_Qwen_Qwen3-VL-4B-Instruct/Figures/UQ/20260320_051336/score/test_predictions.json \
               --out_dir /gscratch/tkishore/MLLM-Judge/Experiments/eval/Qwen_Qwen3-VL-4B-Instruct/   --top_k_score 5   --top_k_pair 3  --task_type score --model_name Qwen3-VL-4B-Instruct

python eval.py --cal_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_Qwen_Qwen3-VL-4B-Instruct/Figures/UQ/20260320_051524/pair/cal_predictions.json \
               --test_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_Qwen_Qwen3-VL-4B-Instruct/Figures/UQ/20260320_051524/pair/test_predictions.json \
               --out_dir /gscratch/tkishore/MLLM-Judge/Experiments/eval/Qwen_Qwen3-VL-4B-Instruct/   --top_k_score 5   --top_k_pair 3  --task_type pair --model_name Qwen3-VL-4B-Instruct

python eval.py --cal_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_Qwen_Qwen3-VL-4B-Instruct/Figures/UQ/20260320_051710/batch/cal_predictions.json \
               --test_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_Qwen_Qwen3-VL-4B-Instruct/Figures/UQ/20260320_051710/batch/test_predictions.json \
               --out_dir /gscratch/tkishore/MLLM-Judge/Experiments/eval/Qwen_Qwen3-VL-4B-Instruct/   --top_k_score 5   --top_k_pair 3  --task_type batch --model_name Qwen3-VL-4B-Instruct








#gemma-3-27b-it
python eval.py --cal_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_google_gemma-3-27b-it/Figures/UQ/20260320_055329/score/cal_predictions.json \
               --test_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_google_gemma-3-27b-it/Figures/UQ/20260320_055329/score/test_predictions.json \
               --out_dir /gscratch/tkishore/MLLM-Judge/Experiments/eval/google_gemma-3-27b-it/   --top_k_score 5   --top_k_pair 3  --task_type score --model_name gemma-3-27b-it

python eval.py --cal_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_google_gemma-3-27b-it/Figures/UQ/20260320_055448/pair/cal_predictions.json \
               --test_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_google_gemma-3-27b-it/Figures/UQ/20260320_055448/pair/test_predictions.json \
               --out_dir /gscratch/tkishore/MLLM-Judge/Experiments/eval/google_gemma-3-27b-it/   --top_k_score 5   --top_k_pair 3  --task_type pair --model_name gemma-3-27b-it

python eval.py --cal_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_google_gemma-3-27b-it/Figures/UQ/20260320_055602/batch/cal_predictions.json \
               --test_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_google_gemma-3-27b-it/Figures/UQ/20260320_055602/batch/test_predictions.json \
               --out_dir /gscratch/tkishore/MLLM-Judge/Experiments/eval/google_gemma-3-27b-it/   --top_k_score 5   --top_k_pair 3  --task_type batch --model_name gemma-3-27b-it


#gemma-3-12b-it
python eval.py --cal_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_google_gemma-3-12b-it/Figures/UQ/20260320_053657/score/cal_predictions.json \
               --test_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_google_gemma-3-12b-it/Figures/UQ/20260320_053657/score/test_predictions.json \
               --out_dir /gscratch/tkishore/MLLM-Judge/Experiments/eval/google_gemma-3-12b-it/   --top_k_score 5   --top_k_pair 3  --task_type score --model_name gemma-3-12b-it

python eval.py --cal_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_google_gemma-3-12b-it/Figures/UQ/20260320_053804/pair/cal_predictions.json \
               --test_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_google_gemma-3-12b-it/Figures/UQ/20260320_053804/pair/test_predictions.json \
               --out_dir /gscratch/tkishore/MLLM-Judge/Experiments/eval/google_gemma-3-12b-it/   --top_k_score 5   --top_k_pair 3  --task_type pair --model_name gemma-3-12b-it

python eval.py --cal_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_google_gemma-3-12b-it/Figures/UQ/20260320_053901/batch/cal_predictions.json \
               --test_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_google_gemma-3-12b-it/Figures/UQ/20260320_053901/batch/test_predictions.json \
               --out_dir /gscratch/tkishore/MLLM-Judge/Experiments/eval/google_gemma-3-12b-it/   --top_k_score 5   --top_k_pair 3  --task_type batch --model_name gemma-3-12b-it


#gemma-3-4b-it
python eval.py --cal_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_google_gemma-3-4b-it/Figures/UQ/20260320_052217/score/cal_predictions.json \
               --test_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_google_gemma-3-4b-it/Figures/UQ/20260320_052217/score/test_predictions.json \
               --out_dir /gscratch/tkishore/MLLM-Judge/Experiments/eval/google_gemma-3-4b-it/   --top_k_score 5   --top_k_pair 3  --task_type score --model_name gemma-3-4b-it

python eval.py --cal_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_google_gemma-3-4b-it/Figures/UQ/20260320_052324/pair/cal_predictions.json \
               --test_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_google_gemma-3-4b-it/Figures/UQ/20260320_052324/pair/test_predictions.json \
               --out_dir /gscratch/tkishore/MLLM-Judge/Experiments/eval/google_gemma-3-4b-it/   --top_k_score 5   --top_k_pair 3  --task_type pair --model_name gemma-3-4b-it

python eval.py --cal_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_google_gemma-3-4b-it/Figures/UQ/20260320_052418/batch/cal_predictions.json \
               --test_json /gscratch/tkishore/MLLM-Judge/Experiments/logs/UQ/slurm_52762797_20260320_051020_google_gemma-3-4b-it/Figures/UQ/20260320_052418/batch/test_predictions.json \
               --out_dir /gscratch/tkishore/MLLM-Judge/Experiments/eval/google_gemma-3-4b-it/   --top_k_score 5   --top_k_pair 3  --task_type batch --model_name gemma-3-4b-it