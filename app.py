from flask import Flask, request, jsonify
from google.cloud import firestore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set path to your service account key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Initialize Firestore client
db = firestore.Client()

# Fetch projects from Firestore
def fetch_projects_from_firestore():
    collection_ref = db.collection('Projects')
    docs = collection_ref.stream()
    project_list = []

    for doc in docs:
        data = doc.to_dict()
        data['id'] = doc.id
        project_list.append(data)

    return project_list

# Compute cosine similarity
def compute_cosine_similarity(user_interests, domains, tools):
    all_texts = [' '.join(user_interests), ' '.join(domains), ' '.join(tools)]

    if not any(text.strip() for text in all_texts):
        return 0.0

    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        return similarity_matrix.mean()
    except ValueError as e:
        print("TF-IDF failed:", e)
        return 0.0

# Recommend projects
def recommend_projects(user_interests):
    user_interests = [i.lower() for i in user_interests]
    projects = fetch_projects_from_firestore()
    recommendations = []

    for project in projects:
        domains = str(project.get('Domain', '')).lower().split(', ')
        tools = str(project.get('Tools', '')).lower().split(', ')
        score = compute_cosine_similarity(user_interests, domains, tools)

        if score > 0:
            recommendations.append({
                'id': project['id'],
                'name':project.get('Name',''),
                'title': project.get('Title', ''),
                'description': project.get('Describe', ''),
                'domain': project.get('Domain', ''),
                'tools': project.get('Tools', ''),
                'similarity_score': score
            })

    recommendations = sorted(recommendations, key=lambda x: x['similarity_score'], reverse=True)
    return recommendations

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    interests = data.get('interests', [])

    if not interests:
        return jsonify({"posts": []})

    recommendations = recommend_projects(interests)
    return jsonify({"posts": recommendations})

if __name__ == '__main__':
    app.run(debug=True)
