
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
Due to time constraints the easiest form of chunking was chosen: namely making each txt file its separate chunk. This made sense as the model used is able to keep the entire file in context. However, this is far from optimal. Alternative possible methods listed below: 

- 

##Assumptions##



##Known Limitations##



##With more time##


##Setup and Run Instructions


