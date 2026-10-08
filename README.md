
## Approach

The approach included: 
1) Generation of knowledge base using "QdrantVectorStore"
2) Application of simple similarity search for the initial retrieval base function
3) Usage of a cross encoder reranker to rerank retrieved results for better context retrieval
4) Creation of a simple frontend to interact with the system
5) Creation of evaluation pipeline 

## Key decisions

Knowledge database: 
This QdrantVectorStore was used due to being relatively lightweight as well as being able to persist across sessions. Originally in-memory database was considered, however, it was deemed too lightweight as the process of creating the store must be repeated each time the RAG system is run.

Chunking: 
Due to time constraints the a default form of chunking was chosen: namely a chunk size of 1000 and overlap of 150 ccharacters was chosen. This is a small to medium chunk size. This has a couple of advantages, namely: easy and fast to build and fits into the embedder. However, it does sacrifice length of context for ease of use. As such, may result in lower accuracy on questions. 


Alternative possible methods listed below: 
- 'Labelled' chunks: take the header and/or the information from the "manifest.json" containing metadata. Append critical details to the start of each chunk. Chunk based off these constructed chunks. 
- Larger chunk sizes and overlap

Reranker: 
Cross embedding reranker was used. This is the more computationally intensive rerankers avaialble. This was chosen due to ease of coding it up. This has negatively affected run speed. To combat this, only the top 4 chunks were returned as part of the RAG. 

It would be optimal to use a different reranker or even forgo this completely (depending on performance on evaluation metrics) in order to increase the number of chunks returned as part of the retrieval. 

Evaluation Metrics and Approach: 
The evaluation approach was to feed "golden" samples to the pipeline and evaluate the responses. The approach was to use an already pre-written package to speed up evaluation. As such, Deepeval was selected. Specifically the following metrics were used:
- Contextual Recall*: how much of the golden answer can be found in the retrieved context (is the information needed to answer actually retrieved?)
- Faithfullness*: the share of claims in the generated answer that are supported by the retrieved context
- Contextual Precision*: whether the relevant retrieved chunks are ranked above the irrelevant ones
- Answer Relevancy*: how well the generated answer addresses the question asked
- Answer Similarity: cosine similarity between the embeddings of the generated answer and the golden answer
- Hit at K: 1 if at least one of the top K retrieved chunks comes from a gold call, otherwise 0 (averaged over questions)
- Reciprocal Rank: 1 / rank of the first retrieved chunk from a gold call (1 = first, 0.5 = second, 0 = not retrieved)

The measures with asterisks are LLM-as-a-judge metrics which come out of the box from deepeval. Answer similarity, Hit at K and Reciporal Rank were coded up manually. The hope is to both validate the retrieval aspect and the generation aspect to ensure both areas are working properly. Contextual Recall and Precision target specifically the contextual accuracy segment. Hit at K and Reciporcal Rank are deterministic functions to test retrieval precision. 

Answer relevancy, similarity and faithfullness check the quality of the generation aspect. 

A quick look at the results json indicate that retrieval seems to work relatively well: relevancy is generally high and hit at k is always 1( indicating that the retrieved chunks are always in the gold chunk). However, it should be noted that this is likely biased upwards as the returned chunk is the entire text document. Notable contextual recall performance is poor, indicating that the retrieved context is often not aligned with the golden answer. Notably this may indiate that the quality of the generated goldens data may be poor. Generation component seems to perform well in tests, high faithfulness indicates claims can be mostly backed up by the returned context. However, it can be noted that, in certain test cases, performance seems to be lower (0.66% minimum) indicating potentailly some hallucinations are occurring. Answer relevance is otherwise high and similarity score (with golden answer) is high.  

Notably, a key limitation is the fact taht the data is syntehtically generated using LLMs. This data has also not been checked over adequately by myself due to time limitations, leading to risk that the evaluation set may be sub-optimal. 

## Assumptions
- Goldens are constructed using copilot. This is potentially inaccurate. I have not been able to verify each fo these goldens due to time constraints. The small number I did test did not have any issues. The implication of this is that the evaluation is inaccurate. Any changes and iterations may also be evaluated inaccurately. This is a fairly critical feature to overcome first as performance on these tests should drive any future changes in the application.
- I have assumed no tables and non-textual data are within the calls dataset. Although in a quick scan of the available data, I did not see any non-textual data (eg: tables or images), I have not had time to verify this. If there were any non-text data, this would imply that generation of the knowlege base and vector store may be inaccurate as the non-text data may not be properly represented with just embeddings. 


## Known Limitations
- Reranker increases runtime substantially as well as forcing the limitation of top results retrieved to only 4. This was originally done due to time constraints - once the code was written and run, didn't have much time to change it over, especially after evaluation was already performed.  
- Simple chunking strategy is not optimal and may decrease accuracy. 
- Lack of proper pydantic output parsing. Simple stringoutputparser has been used. Should ideally want an enforced JSON basemodel with fields for the string answer plus any reference strings and reference files/chunks used
- Underdeveloped testing approach, specifically: 
    - Overreliance on black-box LLM-as-a-judge metrics. Although widely used, it is difficult to know exactly how asterisked LLMas a judge metrics are computed. As such, it is difficult to gauge exactly what is causing specific metrics to take the value they take. Furthermore, there is no benchmarking carried out - there is not indication for what is a "good" score vs a "bad" one. 
    - Unchecked synthetically generated goldens dataset
    - Limited checks for answer accuracy: There is no specific check for answer accuracy. This is a critical flaw as this is arguably the most important metric. Although a quick and dirty metric such as answer ssimilarity metric, this is not a replacement for such a metric. One method is to use a criteria-based LLM-as-a-judge with very clear instructions to pass/fail an asnwer based off criteria when comparing against the golden answer
    - Same LLM model used as llm as judge and chatbot.
- Current knowledge database (QdrantVectorStore) only supports one running app at a time. Usually this is fine, however, if you are running a streamlit app alongside evaluation, this can overlap and error. Would be good to explore other options to see what would be a suitable alternative. 


## With more time
- Firstly: properly develop the testing pipeline so that it can give a clear and concrete view of exactly what the limitations and strengths of the system are. 
- Change chunking strategy to something more sophisticated. 
- Experiment with chunking parameters to see best perofrmance. 
- Experiment with different rerankers. What is required will likely be driven by the change in chunking strategy
- Implement logging


## Setup and Run Instructions

Developed on Windows (PowerShell) with Python 3.11. Commands are run from the project root.

1) Create and activate a virtual environment, then install dependencies:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
(`uv pip install -r requirements.txt` also works.) The first run downloads the local models (`BAAI/bge-m3`, `Qdrant/bm25`, `BAAI/bge-reranker-base`), so it needs internet access and several GB of disk space.

2) Create a `.env` file in the project root containing a Google AI Studio API key (used for Gemini, for both the chatbot and the evaluation judge):
```
GOOGLE_API_KEY=your-key-here
```
Optional: `LLM_MODEL` (default `gemini-2.5-flash`). Langfuse keys (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`) are optional and only enable tracing.

3) Build the vector store from the call transcripts in `data/calls` (writes to `qdrant_db/`; rerunning rebuilds it from scratch):
```powershell
python -m scripts.ingest
```

4) Launch the chatbot:
```powershell
streamlit run app.py
```

5) (Optional) Run the evaluation. Results are written to `eval/results.json`:
```powershell
python -m eval.run_eval
```

Notes:
- The local Qdrant store allows only one process at a time. Stop the Streamlit app (and close any notebook using the store) before rebuilding the store or running the evaluation.
- Chunking and retrieval settings (chunk size, number of retrieved/reranked chunks, models) are in `src/config.py`.


