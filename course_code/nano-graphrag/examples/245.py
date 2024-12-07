import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import ray
import torch
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

from openai import OpenAI
import bz2
import json
import os
from datetime import datetime
import argparse

from loguru import logger
from tqdm.auto import tqdm

from tqdm import tqdm
from nano_graphrag import GraphRAG, QueryParam
import os
import logging
import ollama
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
WORKING_DIR = "./graphrag_cache"

#### CONFIG PARAMETERS ---


logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

# !!! qwen2-7B maybe produce unparsable results and cause the extraction of graph to fail.
MODEL = "llama3.2_3b:ctx32k"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_MODEL_DIM = 768
EMBEDDING_MODEL_MAX_TOKENS = 8192

def load_data_in_batches(dataset_path, batch_size, split=-1):
    """
    Generator function that reads data from a compressed file and yields batches of data.
    Each batch is a dictionary containing lists of interaction_ids, queries, search results, query times, and answers.

    Args:
    dataset_path (str): Path to the dataset file.
    batch_size (int): Number of data items in each batch.

    Yields:
    dict: A batch of data.
    """

    def initialize_batch():
        """ Helper function to create an empty batch. """
        return {"interaction_id": [], "query": [], "search_results": [], "query_time": [], "answer": []}

    try:
        with bz2.open(dataset_path, "rt") as file:
            batch = initialize_batch()
            for line in file:
                try:
                    item = json.loads(line)

                    if split != -1 and item["split"] != split:
                        continue

                    for key in batch:
                        batch[key].append(item[key])

                    if len(batch["query"]) == batch_size:
                        yield batch
                        batch = initialize_batch()
                except json.JSONDecodeError:
                    logger.warn("Warning: Failed to decode a line.")
            # Yield any remaining data as the last batch
            if batch["query"]:
                yield batch
    except FileNotFoundError as e:
        logger.error(f"Error: The file {dataset_path} was not found.")
        raise e
    except IOError as e:
        logger.error(f"Error: An error occurred while reading the file {dataset_path}.")
        raise e


def generate_predictions(dataset_path, model, split,output_directory):
    """
    Processes batches of data from a dataset to generate predictions using a model.

    Args:
    dataset_path (str): Path to the dataset.
    model (object): UserModel that provides `get_batch_size()` and `batch_generate_answer()` interfaces.

    Returns:
    tuple: A tuple containing lists of queries, ground truths, and predictions.
    """
    queries, ground_truths, predictions = [], [], []
    batch_size = model.get_batch_size()

    for batch_index, batch in enumerate(tqdm(load_data_in_batches(dataset_path, batch_size, split), desc="Generating predictions")):
        batch_ground_truths = batch.pop("answer")  # Remove answers from batch and store them
        if batch_index<19:
            continue
        batch_predictions = model.batch_generate_answer(batch)
        # import pdb; pdb.set_trace()
        queries.extend(batch["query"])
        ground_truths.extend(batch_ground_truths)
        predictions.extend(batch_predictions)

        # Save the entire batch as a JSON file
        batch_data = {
            "queries": batch["query"],
            "ground_truths": batch_ground_truths,
            "predictions": batch_predictions
        }
        file_path = os.path.join(output_directory, f"batch_prediction_{batch_index}.json")
        with open(file_path, "w") as f:
            json.dump(batch_data, f, indent=4)

    return queries, ground_truths, predictions

# We're using Ollama to generate embeddings for the BGE model
@wrap_embedding_func_with_attrs(
    embedding_dim=EMBEDDING_MODEL_DIM,
    max_token_size=EMBEDDING_MODEL_MAX_TOKENS,
)
async def ollama_embedding(texts: list[str]) -> np.ndarray:
    embed_text = []
    for text in texts:
        data = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        embed_text.append(data["embedding"])

    return embed_text
async def ollama_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # remove kwargs that are not supported by ollama
    kwargs.pop("max_tokens", None)
    kwargs.pop("response_format", None)

    ollama_client = ollama.AsyncClient()
    messages = []
            
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    # import pdb; pdb.set_trace()
    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------
    response = await ollama_client.chat(model=MODEL, messages=messages, **kwargs)

    result = response["message"]["content"]
    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": MODEL}})
    # -----------------------------------------------------
    return result
# Define the number of context sentences to consider for generating an answer.
NUM_CONTEXT_SENTENCES = 20
# Set the maximum length for each context sentence (in characters).
MAX_CONTEXT_SENTENCE_LENGTH = 1000
# Set the maximum context references length (in characters).
MAX_CONTEXT_REFERENCES_LENGTH = 4000

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
AICROWD_SUBMISSION_BATCH_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# VLLM Parameters 
VLLM_TENSOR_PARALLEL_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# Sentence Transformer Parameters
SENTENTENCE_TRANSFORMER_BATCH_SIZE = 32 # TUNE THIS VARIABLE depending on the size of your embedding model and GPU mem available

def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)

class GraphRAGModel:
    def __init__(self, llm_name="meta-llama/Llama-3.2-3B-Instruct", is_server=False, vllm_server=None):
        pass
    def convert_html_to_text(self, html_source):
        soup = BeautifulSoup(html_source, "lxml")
        text = soup.get_text(" ", strip=True)
        if not text:
            return ""
        return text

    def insert_chunks_to_graphrag(self, chunks,rag):
        for chunk in chunks:
            import pdb; pdb.set_trace()
            

    
    
    def get_batch_size(self) -> int:
        self.batch_size = AICROWD_SUBMISSION_BATCH_SIZE
        return self.batch_size

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        from time import time

        # import pdb; pdb.set_trace()
        remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
        remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
        remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
        remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
        remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")

        rag = GraphRAG(
            working_dir=WORKING_DIR,
            enable_llm_cache=True,
            best_model_func=ollama_model_if_cache,
            cheap_model_func=ollama_model_if_cache,
            embedding_func=ollama_embedding,

        )
        
        start = time()
        print("indexing time:", time() - start)
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]
        # import pdb; pdb.set_trace()
        chunks = [[self.convert_html_to_text(html_source_i["page_result"]) for html_source_i in html_source] for html_source in batch_search_results]
        for chunk in chunks:
            # import pdb; pdb.set_trace()
            num_str = 0
            for chunk_i in chunk:
                num_str += len(chunk_i)
            print(f"num_str: {num_str}")
            # import pdb; pdb.set_trace()
            rag.insert(chunk)
            # self.insert_chunks_to_graphrag(chunk,rag)

        batch_retrieval_results = []
        answers = []
        del rag
        rag = GraphRAG(
            working_dir=WORKING_DIR,
            best_model_func=ollama_model_if_cache,
            cheap_model_func=ollama_model_if_cache,
            embedding_func=ollama_embedding,

        )
        for query in queries:
            
            system_prompt ="You are provided with a question and various references. Your task is to answer the question succinctly, using the fewest words possible. If the references do not contain the necessary information to answer the question, respond with 'I don't know'. The question is: "
          
            query = system_prompt+query
            response = rag.query(
                query, param=QueryParam(mode="local")
            )
            
            answers.append(response)
        remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
        remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
        remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
        remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
        remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")
        del rag
        return answers

    def format_prompts(self, queries, query_times, batch_retrieval_results):
        system_prompt = (
            "You are provided with a question and various references. "
            "Your task is to answer the question succinctly, using the fewest words possible. "
            "If the references do not contain the necessary information to answer the question, respond with 'I don't know'."
        )
        formatted_prompts = []

        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]

            user_message = ""
            references = ""

            if len(retrieval_results) > 0:
                references += "# References \n"
                for snippet in retrieval_results:
                    references += f"- {snippet.strip()}\n"

            references = references[:MAX_CONTEXT_REFERENCES_LENGTH]

            user_message += f"{references}\n------\n\n"
            user_message += "Using only the references listed above, answer the following question:\n"
            user_message += f"Current Time: {query_time}\n"
            user_message += f"Question: {query}\n"

            if self.is_server:
                formatted_prompts.append(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ]
                )
            else:
                formatted_prompts.append(
                    self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                )

        return formatted_prompts
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, default="/root/crag/crag_task_1_dev_v4_release.jsonl.bz2",
                        choices=["example_data/dev_data.jsonl.bz2", # example data
                                 "data/crag_task_1_dev_v4_release.jsonl.bz2", # full data
                                 ])
    parser.add_argument("--split", type=int, default=-1,
                        help="The split of the dataset to use. This is only relevant for the full data: "
                             "0 for public validation set, 1 for public test set")

    parser.add_argument("--model_name", type=str, default="vanilla_baseline",
                        choices=["vanilla_baseline",
                                 "rag_baseline",
                                 "graphrag"# add your model here
                                 ],
                        )


    args = parser.parse_args()

    dataset_path = args.dataset_path
    dataset = dataset_path.split("/")[0]
    split = -1
    if dataset == "data":
        split = args.split
        if split == -1:
            raise ValueError("Please provide a valid split value for the full data: "
                             "0 for public validation set, 1 for public test set.")
    dataset_path = os.path.join("..", dataset_path)

    
    
    model = GraphRAGModel()
    
    # make output directory
    output_directory = os.path.join("..", "output", dataset, 'graphrag', 'llama3.2')
    os.makedirs(output_directory, exist_ok=True)

    # Generate predictions
    queries, ground_truths, predictions = generate_predictions(dataset_path, model, split,output_directory)

    # save predictions
    json.dump({"queries": queries, "ground_truths": ground_truths, "predictions": predictions},
              open(os.path.join(output_directory, "predictions.json"), "w"), indent=4)
