Reference from  LLMzoomcamp 2024
https://github.com/DataTalksClub/llm-zoomcamp/

query : "I just discovered the course. Can I still join?"

Relevant docs: doc1

for each record in  FAQ:
    generate 5 questions

1000 ->  5000

for each q in ground truth dataset:
    execute q
    check if d is in results

steps for elastic search :
1. get elastic search client
"""
from elasticsearch import Elasticsearch

es_client = Elasticsearch('http://localhost:9200') 

index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"},
            "id": {"type": "keyword"},
            "question_text_vector": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine"
            },
        }
    }
}
"""
2. delete if any and create new indicies
"""
index_name = "course-questions"

es_client.indices.delete(index=index_name, ignore_unavailable=True)
es_client.indices.create(index=index_name, body=index_settings)
""" 
3. get embedding for the "question_text_vector" vector
"""
for doc,em in zip(tqdm(documents),embeddings):
    doc['question_text_vector'] = em
"""
4. Index the documents:
"""
for doc in tqdm(documents):
    es_client.index(index=index_name, document=doc)
"""
5.  Elastic search for the query and get the results 
"""
def elastic_search_knn(field, vector, course):
    knn = {
        "field": field,
        "query_vector": vector,
        "k": 5,
        "num_candidates": 10000,
        "filter": {
            "term": {
                "course": course
            }
        }
    }

    search_query = {
        "knn": knn,
        "_source": ["text", "section", "question", "course", "id"]
    }

    es_results = es_client.search(
        index=index_name,
        body=search_query
    )
    
    result_docs = []
    
    for hit in es_results['hits']['hits']:
        result_docs.append(hit['_source'])

    return result_docs
    """

6. From each query get the elastic search results 
"""
def question_vector_knn(q):
    question = q['question']
    course = q['course']

    v_q = embedding_model.encode(question)

    return elastic_search_knn('question_text_vector', v_q, course)
    """