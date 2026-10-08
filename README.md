
##Approach##

The approach included: 
1) Generation of knowledge base using "QdrantVectorStore"
2) Application of simple similarity search for the initial retrieval base function
3) Usage of a cross encoder reranker to rerank retrieved results for better context retrieval
4) Creation of a simple frontend to interact with the system 

##Key decisions##

Knowledge database: 
This QdrantVectorStore was used due to being relatively lightweight as well as being able to persist across sessions. Originally in-memory database was considered, however, it was deemed too lightweight as the process of creating the store must be repeated each time the RAG system is run.

Chunking: 
Due to time constraints the easiest form of chunking was chosen: namely making each txt file its separate chunk. This made sense as the model used is able to keep the entire file in context. However, this is far from optimal. Noted drawbacks include:
- Substantially increased runtime for both creation of vector store
- Substantially increased runtime for queries on RAG database due to large chunk being entered into conext
- Possible decrease in accuracy of answers due to overloading the context window 
- Difficulty validating relevance of returned context - refereences returned are for an entire text file so metrics assessing relevance can only check if relevant information is within that particular text file
- 

Alternative possible methods listed below: 
- 'Labelled' chunks: take the header and/or the information from the "manifest.json" containing metadata. Append critical details to the start of each chunk. Chunk based off these constructed chunks. 
- 

Reranker: 
Cross embedding reranker was used. This is the most computationally

Base Retriever:


Evaluation Metrics and Approach: 
The evaluation approach was to feed "golden" samples to the pipeline and evaluate the responses. Specifically the following metrics were used:
- Contextual Recall*
- Faithfullness*
- Precision*
- Answer Relevancy*
- Answer Similarity
- Hit at K 
- Reciprocal Rank

The measures with asterisks are LLM-as-a-judge metrics which come out of the box from deepeval.

##Assumptions##
- I have assumed no tables and non-textual data are within the calls dataset. Although in a quick scan of the available data, I did not see any non-textual data, I have not had time to verify this. 
- Goldens are constructed using copilot. This is potentially inaccurate. I have not been able to verify each fo these goldens due to time constraints. This 



##Known Limitations##



##With more time##


##Setup and Run Instructions


