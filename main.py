import typer
import numpy
import os
from sentence_transformers import SentenceTransformer
import httpx
import gzip
from pathlib import Path

############################################################################################################################################################################################################################################


model = SentenceTransformer(
    "mixedbread-ai/mxbai-embed-large-v1",
    prompts={"retrieval": "Represent this sentence for searching relevant passages: "},
)

############################################################################################################################################################################################################################################


URL= "https://north-1.cloud.snic.se:8080/swift/v1/AUTH_d9d5ac98cb2b4a3091b60040077e8efc/plantgenie-knowledge/Picab02_230926_at01_longest_representative_annotations_merged_sorted_non_redundant_panthers.tsv.gz"

p = Path("/home/seva/Typer/semantic-query/Picab02_230926_at01_longest_representative_annotations_merged_sorted_non_redundant_panthers.tsv")

if not p.exists():
    response = httpx.get(URL, follow_redirects=True)
    content = gzip.decompress(response.content)
    p.write_bytes(content)


############################################################################################################################################################################################################################################
text = p.read_text()
lines = text.splitlines()

ID = []
description = [] 


for line in lines:
    columns = line.split("\t")
    ID.append(columns[0])
    description.append(columns[4])

# print(f"\n\nThese are the first 10 lines of IDs with their description: {list(zip(ID, description))[:10]} \n\n")

############################################################################################################################################################################################################################################



gene_ids_file = "gene-ids.npy"
embeddings_file = "annotations-embeddings.npy"

if os.path.exists(embeddings_file): 
    # print("Loading embeddings from disk...")
    embeddings = numpy.load(embeddings_file)
    gene_ids = list(numpy.load(gene_ids_file, allow_pickle=True))
else:
    print(f"Computing embeddings for {len(description)} gene descriptions...")
    embeddings = model.encode(description, show_progress_bar=True)
    numpy.save(embeddings_file, embeddings)
    numpy.save(gene_ids_file, numpy.array(ID, dtype=object))
    print("Embeddings saved to disk.")
# print(f"Embedding matrix shape: {embeddings.shape}")



############################################################################################################################################################################################################################################


# def search(query, num_of_top_results):
#     """ Search for semantically similar genes based on a query with a desired number of top results to appear. """
#     query_embedding = model.encode_query(query, prompt= "Search: ")                             
#     similarities = model.similarity(query_embedding, embeddings).squeeze()                        
#     indices= similarities.argsort(descending=True).squeeze().tolist()                               
#     print(f"\nTop {num_of_top_results} results for \"{query}\": ")
#     for i in range(num_of_top_results):                                                          
#         index = indices[i]
#         score = similarities[index].item()
#         print(f"\nGene: {gene_ids[index]} \nIndices: {embeddings[index]} \nSimilarity score: {score} \nDescription: {description[index]}")




############################################################################################################################################################################################################################################

app = typer.Typer()

@app.command()
def search(query: str, num_of_top_results:int = 10):
    """ Search for semantically similar genes based on a query with a desired number of top results to appear. Automatically prints top 10 results unless specified. """
    query_embedding = model.encode_query(query, prompt= "Search: ")                             
    similarities = model.similarity(query_embedding, embeddings).squeeze()                        
    indices= similarities.argsort(descending=True).squeeze().tolist()                               
    print(f"\nTop {num_of_top_results} results for \"{query}\": ")
    for i in range(num_of_top_results):                                                          
        index = indices[i]
        score = similarities[index].item()
        print(f"\nGene: {gene_ids[index]} \nIndices: {embeddings[index]} \nSimilarity score: {score} \nDescription: {description[index]}")





############################################################################################################################################################################################################################################



@app.command()
def isearch():
    """Interactive search to find genes and show number of desired results"""
    iquery = input("What are you looking for? ")
    inumber= int(input("How many results do you want to see? "))
    search(iquery, inumber)
    while True: 
        iquery = input("\n\nDo you want to search for another set of genes? ")
        for tries in range(100):
            if iquery.lower() in ["no", "quit", "nah"]:
                print("Okay, see you next time!")
                break 
            elif iquery.lower() in  ["yes", "sure", "yeah", "ye"]:  
                iquery= input ("What kind of genes are you looking for? ")
                inumber= input("How many results would you like to see? ")
                search(iquery, int(inumber)) 
                iquery = input("\n\nDo you want to search for another set of genes? ")
            else:
                iquery = input("I'm sorry, I didn't understand this. If you want to continue type yes, if you want to quit type quit. ").lower()
                tries +=1
        else: 
            break
        if iquery.lower() in ["no", "quit", "nah"]:
                break 
        else:
            continue
            
                

############################################################################################################################################################################################################################################


if __name__ == "__main__": 
    app()

