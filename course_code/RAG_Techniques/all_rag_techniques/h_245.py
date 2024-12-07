import asyncio
import os
import sys
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
from langchain_openai import ChatOpenAI
from langchain.chains.summarize.chain import load_summarize_chain
from langchain.docstore.document import Document
from langchain.docstore.document import Document

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..'))) # Add the parent directory to the path sicnce we work with notebooks
from helper_functions import *
from evaluation.evalute_rag import *
from helper_functions import encode_pdf, encode_from_string
import bz2
from loguru import logger
from bs4 import BeautifulSoup
import vllm
from tqdm import tqdm
VLLM_TENSOR_PARALLEL_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.5 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
MAX_CONTEXT_REFERENCES_LENGTH = 4000

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
def convert_html_to_text(html_source):
        soup = BeautifulSoup(html_source, "lxml")
        text = soup.get_text(" ", strip=True)
        if not text:
            return ""
        return text
async def batch_generate_db(batch, chunk_size=1000, chunk_overlap=200, is_string=False):
    batch_interaction_ids = batch["interaction_id"]
    queries = batch["query"]
    batch_search_results = batch["search_results"]
    query_times = batch["query_time"]
    # import pdb; pdb.set_trace()
    chunks = [[convert_html_to_text(html_source_i["page_result"]) for html_source_i in html_source] for html_source in batch_search_results]
    # Create document-level summaries
    batch_size = 5  # Adjust this based on your rate limits
    documents = []

    for c_i,chunk in enumerate(chunks[0]):
        doc =  Document(page_content=chunk, metadata={"source": "local", "page": c_i})
        documents.append(doc)

    summary_llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)
    summary_chain = load_summarize_chain(summary_llm, chain_type="map_reduce")
    
    async def summarize_doc(doc):
        """
        Summarizes a single document with rate limit handling.
        
        Args:
            doc: The document to be summarized.
            
        Returns:
            A summarized Document object.
        """
        # Retry the summarization with exponential backoff
        summary_output = await retry_with_exponential_backoff(summary_chain.ainvoke([doc]))
        summary = summary_output['output_text']
        return Document(
            page_content=summary,
            metadata={"source": 'local', "page": doc.metadata["page"], "summary": True}
        )

    # Process documents in smaller batches to avoid rate limits
    
    summaries = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        batch_summaries = await asyncio.gather(*[summarize_doc(doc) for doc in batch])
        summaries.extend(batch_summaries)
        await asyncio.sleep(1)  # Short pause between batches

    # Split documents into detailed chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    detailed_chunks = await asyncio.to_thread(text_splitter.split_documents, documents)

    # Update metadata for detailed chunks
    for i, chunk in enumerate(detailed_chunks):
        chunk.metadata.update({
            "chunk_id": i,
            "summary": False,
            "page": int(chunk.metadata.get("page", 0))
        })

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Create vector stores asynchronously with rate limit handling
    async def create_vectorstore(docs):
        """
        Creates a vector store from a list of documents with rate limit handling.
        
        Args:
            docs: The list of documents to be embedded.
            
        Returns:
            A FAISS vector store containing the embedded documents.
        """
        return await retry_with_exponential_backoff(
            asyncio.to_thread(FAISS.from_documents, docs, embeddings)
        )

    # Generate vector stores for summaries and detailed chunks concurrently
    summary_vectorstore, detailed_vectorstore = await asyncio.gather(
        create_vectorstore(summaries),
        create_vectorstore(detailed_chunks)
    )

    return summary_vectorstore, detailed_vectorstore
        
def retrieve_hierarchical(query, summary_vectorstore, detailed_vectorstore, k_summaries=3, k_chunks=5):
    """
    Performs a hierarchical retrieval using the query.

    Args:
        query: The search query.
        summary_vectorstore: The vector store containing document summaries.
        detailed_vectorstore: The vectaor store containing detailed chunks.
        k_summaries: The number of top summaries to retrieve.
        k_chunks: The number of detailed chunks to retrieve per summary.

    Returns:
        A list of relevant detailed chunks.
    """
    
    # Retrieve top summaries
    top_summaries = summary_vectorstore.similarity_search(query, k=k_summaries)
    
    relevant_chunks = []
    for summary in top_summaries:
        # For each summary, retrieve relevant detailed chunks
        page_number = summary.metadata["page"]
        page_filter = lambda metadata: metadata["page"] == page_number
        page_chunks = detailed_vectorstore.similarity_search(
            query, 
            k=k_chunks, 
            filter=page_filter
        )
        relevant_chunks.extend(page_chunks)
    
    return relevant_chunks
llm = vllm.LLM(
    model="meta-llama/Llama-3.2-3B-Instruct",
    worker_use_ray=True,
    tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
    trust_remote_code=True,
    dtype="half",  # note: bfloat16 is not supported on nvidia-T4 GPUs
    enforce_eager=True
)
tokenizer = llm.get_tokenizer()
def format_prompts(queries, query_times, batch_retrieval_results=[]):
    """
    Formats queries, corresponding query_times and retrieval results using the chat_template of the model.
        
    Parameters:
    - queries (List[str]): A list of queries to be formatted into prompts.
    - query_times (List[str]): A list of query_time strings corresponding to each query.
    - batch_retrieval_results (List[str])
    """        
    system_prompt = "You are provided with a question and various references. Your task is to answer the question succinctly, using the fewest words possible. If the references do not contain the necessary information to answer the question, respond with 'I don't know'. There is no need to explain the reasoning behind your answers."
    formatted_prompts = []
    # import pdb; pdb.set_trace()
    for _idx, query in enumerate(queries):
        query_time = query_times[_idx]
        retrieval_results = batch_retrieval_results[_idx]
        user_message = ""
        references = ""
        
        if len(retrieval_results) > 0:
            references += "# References \n"
            # Format the top sentences as references in the model's prompt template.
            for _snippet_idx, snippet in enumerate(retrieval_results):
                references += f"- {snippet.strip()}\n"
        
        references = references[:MAX_CONTEXT_REFERENCES_LENGTH]
        # Limit the length of references to fit the model's input size.

        user_message += f"{references}\n------\n\n"
        user_message 
        user_message += f"Using only the references listed above, answer the following question: \n"
        user_message += f"Current Time: {query_time}\n"
        user_message += f"Question: {query}\n"

        
        formatted_prompts.append(
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    return formatted_prompts

async def process_batches(batch_index, batch):
    

    summary_store, detailed_store = await batch_generate_db(batch)
    summary_store.save_local("../vector_stores/summary_store")
    detailed_store.save_local("../vector_stores/detailed_store")
    query = batch["query"]
    query_times = batch["query_time"]
    # import pdb; pdb.set_trace()
    relevant_chunks = retrieve_hierarchical(query[0], summary_store, detailed_store)
    
    # import pdb; pdb.set_trace()
    batch_retrieval_results = [[relevant_chunk_i.page_content for relevant_chunk_i in relevant_chunks]]
    formatted_prompts = format_prompts(query, query_times, batch_retrieval_results)

    responses = llm.generate(
        formatted_prompts,
        vllm.SamplingParams(
            n=1,  # Number of output sequences to return for each prompt.
            top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
            temperature=0.1,  # randomness of the sampling
            skip_special_tokens=True,  # Whether to skip special tokens in the output.
            max_tokens=50,  # Maximum number of tokens to generate per output sequence.
        ),
        use_tqdm=False
    )
    answers = []
    for response in responses:
        answers.append(response.outputs[0].text)

    return answers
    # import pdb; pdb.set_trace()
# queries, ground_truths, predictions = [], [], []
# for batch_index, batch in enumerate(tqdm(load_data_in_batches('/root/crag/crag_task_1_dev_v4_release.jsonl.bz2', 1, -1))):
#     batch_ground_truths = batch.pop("answer")
#     batch_predictions = asyncio.run(process_batches(batch_index, batch))

#     ground_truths.extend(batch_ground_truths)
#     predictions.extend(batch_predictions)
#     queries.extend(batch["query"])
    
# output_directory = os.path.join("..", "output", "hierarchical_indices", "llama3.2")
# os.makedirs(output_directory, exist_ok=True)
# json.dump({"queries": queries, "ground_truths": ground_truths, "predictions": predictions},
#         open(os.path.join(output_directory, "predictions.json"), "w"), indent=4)

async def main():
    # Your asynchronous code here
    queries, ground_truths, predictions = [], [], []
    output_directory='/root/hir'
    for batch_index, batch in enumerate(tqdm(load_data_in_batches('../data/crag_task_1_dev_v4_release.jsonl.bz2', 1, -1))):
        if batch_index < 908:
            continue
        batch_ground_truths = batch.pop("answer")
        
        batch_predictions = await process_batches(batch_index, batch)

        ground_truths.extend(batch_ground_truths)
        predictions.extend(batch_predictions)
        queries.extend(batch["query"])
        file_path = os.path.join(output_directory, f"batch_prediction_{batch_index}.json")
        batch_data = {
            "queries": batch["query"],
            "ground_truths": batch_ground_truths,
            "predictions": batch_predictions
        }
        with open(file_path, "w") as f:
            json.dump(batch_data, f, indent=4)
    output_directory = os.path.join("..", "output", "hierarchical_indices", "llama3.2")
    os.makedirs(output_directory, exist_ok=True)
    json.dump({"queries": queries, "ground_truths": ground_truths, "predictions": predictions},
            open(os.path.join(output_directory, "predictions.json"), "w"), indent=4)

if __name__ == "__main__":
    asyncio.run(main())