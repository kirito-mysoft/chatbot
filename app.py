from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

nlp = spacy.load("en_core_web_sm")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_knowledge_base(file_path):
    knowledge_base = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            question, answer = line.strip().split('|')
            knowledge_base[question] = answer
    return knowledge_base

def log_fallback_question(user_input, log_file='fallback_questions.txt'):
    with open(log_file, 'a', encoding='utf-8') as file:
        file.write(f"{user_input}\n")


def find_best_match(user_input, knowledge_base):
    questions = list(knowledge_base.keys())
    answers = list(knowledge_base.values())
    
    question_embeddings = sentence_model.encode(questions)
    user_input_embedding = sentence_model.encode([user_input])
    
    similarity_scores = cosine_similarity(user_input_embedding, question_embeddings)
    best_match_idx = similarity_scores.argmax()
    
    if similarity_scores.max() > 0.80:
        return answers[best_match_idx]
    else:
        log_fallback_question(user_input)  
        return "দুঃখিত, আমি আপনার প্রশ্ন বুঝতে পারিনি।"

knowledge_base = load_knowledge_base('knowledge_base.txt')


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('user_input', '')
    
    if not user_input:
        return jsonify({"error": "No input provided"}), 400
    
    response = find_best_match(user_input, knowledge_base)
    return jsonify({"user_input": user_input, "response": response})

if __name__ == "__main__":
    app.run(debug=True)
