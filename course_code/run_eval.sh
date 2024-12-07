# export CUDA_VISIBLE_DEVICES=0

python evaluate.py \
  --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
  --model_name "rag_chunk" \
  --pred_save_path "predictions_768_75.json"