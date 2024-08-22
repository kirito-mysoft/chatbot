from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from langchain_openai import OpenAI  # Use this if you have access
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.llms import HuggingFaceLLM  # Import HuggingFaceLLM for GPT-Neo

app = Flask(__name__)

# Initialize models
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Elasticsearch client with hosts parameter
es = Elasticsearch(hosts=["http://localhost:9200"])  # Update with your Elasticsearch host and port

# Initialize LangChain with GPT-Neo
llm = HuggingFaceLLM(model_name="EleutherAI/gpt-neo-125M")  # Use GPT-Neo model

# Create a prompt template in Bangla
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    আপনি একটি এআই সহকারী যিনি প্রদত্ত প্রসঙ্গের উপর ভিত্তি করে সঠিক এবং বিস্তারিত উত্তর প্রদান করেন।

    প্রসঙ্গ:
    {context}

    প্রশ্ন:
    {question}

    উত্তর:
    """
)

# Create a RetrievalQA instance with LangChain
retriever = create_retrieval_chain(llm, prompt_template)

# Retrieve documents from Elasticsearch
def retrieve_documents(query, index_name='knowledge_base'):
    search_query = {
        "query": {
            "match": {
                "question": query
            }
        }
    }
    response = es.search(index=index_name, body=search_query)
    hits = response['hits']['hits']
    return [hit['_source']['answer'] for hit in hits]

# Generate responses using LangChain and GPT-Neo
def generate_response(user_input):
    # Retrieve relevant answers from Elasticsearch
    retrieved_answers = retrieve_documents(user_input)
    
    if retrieved_answers:
        # Combine retrieved answers into a context
        combined_context = " ".join(retrieved_answers)
        
        # Generate response using LangChain with the detailed Bangla prompt
        response = retriever.invoke({
            'context': combined_context,
            'question': user_input
        })
        
        return response
    else:
        return "দুঃখিত, আপনার প্রশ্নের সাথে সম্পর্কিত কোন উত্তর খুঁজে পাওয়া যায়নি।"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('user_input', '')
    
    if not user_input:
        return jsonify({"error": "কোন ইনপুট প্রদান করা হয়নি"}), 400
    
    response = generate_response(user_input)
    return jsonify({"user_input": user_input, "response": response})

if __name__ == "__main__":
    app.run(debug=True)