1. Go to directory `course_code/`
2. Generate or evaluate different models with the following commands:
   ### Vanilla Baseline / RAG Baseline
   Please run the following command to generate from script: 
   ```
   python generate.py \
        --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
        --pred_save_path "{path_to_save_predictions}.json" \
        --split 1 \
        --model_name "vanilla_baseline" / "rag_baseline" \ # Choose the model for generation
        --llm_name "meta-llama/Llama-3.2-3B-Instruct"
    ```

    Please run the following command for evaluation (make sure the --model_name and --pred_save_path are identical with generation command): 
    ```
    python evaluate.py \
        --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
        --model_name "vanilla_baseline" / "rag_baseline" \ # Choose the model for generation
        --pred_save_path "{path_to_save_predictions}.json"
    ```

    Please run the following command to test our prediction files:
    ```
    python evaluate.py \
        --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
        --model_name "vanilla_baseline" / "rag_baseline" \ # Choose the model for generation
        --pred_save_path "predictions.json"
    ```

    ### RAG Chunk
    Please change `CHUNK_SIZE` and `CHUNK_OVERLAP` at line 22 and 23 in `course_code/rag_chunk.py`, and then run the following command to generate from script: 
    ```
    python generate.py \
        --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
        --pred_save_path "{path_to_save_predictions}.json" \
        --split 1 \
        --model_name "rag_chunk" \
        --llm_name "meta-llama/Llama-3.2-3B-Instruct"
    ```

    Please run the following command for evaluation (make sure  --pred_save_path is identical with generation command): 
    ```
    python evaluate.py \
        --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
        --model_name "rag_chunk" \
        --pred_save_path "{path_to_save_predictions}.json"
    ```

    Please run the following command to test our prediction files for specific `CHUNK_SIZE` and `CHUNK_OVERLAP` settings:
    ```
    python evaluate.py \
        --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
        --model_name "rag_chunk" \
        --pred_save_path "predictions_{CHUNK_SIZE}_{CHUNK_OVERLAP}.json"
    ```

    ### Top K
    Modify `NUM_CONTEXT_SENTENCES` for rag\_baseline.py


    ### Prompt
     ```
    python evaluate.py \
        --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
        --model_name "rag_baseline" \
     ```
     ```
    python generate.py \
        --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
        --split 1 \
        --model_name "rag_baseline" \
        --llm_name "meta-llama/Llama-3.2-3B-Instruct"
     ```


    ### Hierarchical Inference
    Please run the following command to generate from script: `python ./RAG_Techniques/all_rag_techniques/h_245.py`
    **NOTE**: The generation script uses relative path to dump predictions file, so please be careful to input the correst --pred_path when evaluating the outputs

    Please run the following command for evaluation:  
    ```
    python evaluate_path.py \
        --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
        --pred_path "../output/hierarchical_indices/llama3.2/predictions.json"
    ```

    Please run the following command to test our prediction files for Hierarchical Inference:
    ```
    python evaluate_path.py \
        --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
        --pred_path "../output/merged_output_hierarchical.json"
    ```

    ### GraphRAG
    Please run the following command to generate from script:
    ```
    python ./nano-graphrag/examples/245.py
        --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
        --split 1 \
        --model_name "graphrag"
    ```
    **NOTE**: The generation script uses relative path to dump predictions file, so please be careful to input the correst --pred_path when evaluating the outputs

    Please run the following command for evaluation: 
    ```
    python evaluate_path.py \
        --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
        --pred_path "../output/data/graphrag/llama3.2/predictions.json'
    ```

    Please run the following command to test our prediction files for GraphRAG: 
    ```
    python evaluate_path.py \
        --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
        --pred_path "../output/graphrag/llama3.2/batch_prediction_19.json"
    ```