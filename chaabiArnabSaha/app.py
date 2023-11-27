from flask import Flask, request, jsonify
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer
from transformers import pipeline


app = Flask(__name__)

client = QdrantClient("localhost", port=6333)
encoder = SentenceTransformer('all-MiniLM-L6-v2')

model_name = "deepset/roberta-base-squad2"

nlp = pipeline("question-answering",model = model_name,tokenizer=model_name)


@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        query = data['query']

        query_vector = encoder.encode([query])

        results = client.search(
        collection_name='products_data',
        query_vector=query_vector.tolist(),
        limit=6)

        context = [f"brand is {dict(val)['payload']['brand']}"+
               f" category is {dict(val)['payload']['category']}"+ 
                f" description is {dict(val)['payload']['description']}"+
                f" market_price is {dict(val)['payload']['market_price']}"+
                f" product is {dict(val)['payload']['product']}"+
                f" rating is {dict(val)['payload']['rating']}"
                for val in results]
        
        context = ". ".join(context)

        input = {
        'question':query,
        'context':context
        }
        answer = nlp(input)["answer"]

        return jsonify({'message': answer})
    
    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == '__main__':
    app.run(debug=True)