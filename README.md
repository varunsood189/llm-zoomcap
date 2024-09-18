# llm-zoomcap
Donot have access to openai, so tried with one of the free services.
I am working on the course work from DataTalksClub/llm-zoomcamp.
Link: https://github.com/DataTalksClub/llm-zoomcamp
parse.ipynb from above link
    clean_line :removes leading and trailing spaces from a line and also strips any Unicode Byte Order Mark (BOM) (\uFEFF) 
    read_faq: read document from doc to json format with text, section and question.
minsearch from https://github.com/alexeygrigorev/minsearch
    fit(docs): Fits the index with a list of documents, transforming the text fields into TF-IDF matrices and storing keyword field data in a  DataFrame.
    search:  Searches the index using a query. Filters and Boosts. Top num_results documents based on cosine similarity ranking.
