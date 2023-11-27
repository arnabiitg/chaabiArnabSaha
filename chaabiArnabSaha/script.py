from qdrant_client import models,QdrantClient
from qdrant_client.http.models import PointStruct
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pandas as pd

# Run this script to upload the data from the csv to the qdrant database by getting their embedddings
client = QdrantClient("localhost", port=6333)

encoder = SentenceTransformer('all-MiniLM-L6-v2')


df = pd.read_csv("bigBasketProducts.csv")
documents = list(df.astype("str").to_dict(orient="records"))

client = QdrantClient("localhost", port=6333)

client.recreate_collection(
    collection_name = "products_data",
    vectors_config =  models.VectorParams(
            size = encoder.get_sentence_embedding_dimension(),
            distance = models.Distance.COSINE,
            on_disk=True
        )
)

for index in tqdm(range(len(documents))):
    document = documents[index]
    client.upsert(
        collection_name="products_data",
        points=[
            PointStruct(
                id=index,
                vector=encoder.encode(
                    'description : ' + document['description'] +
                    ' product : ' +  document['product'] +
                    ' sub_category : '+ document['sub_category']+
                    ' brand : ' + document['brand'] +
                    ' typ : '+ document['type']
                 ).tolist(),
                payload= document,
            )
        ]
    )