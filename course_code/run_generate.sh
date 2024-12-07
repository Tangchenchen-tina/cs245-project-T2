# export CUDA_VISIBLE_DEVICES=0

python generate.py \
  --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
  --pred_save_path "predictions_1280_0.json" \
  --split 1 \
  --model_name "rag_chunk" \
  --llm_name "meta-llama/Llama-3.2-3B-Instruct"