from flask import Flask, request, jsonify
from google.cloud import firestore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Set path to your service account key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your/serviceAccountKey.json"

# Initialize Firestore client
db = firestore.Client()

# Fetch projects from Firestore
def fetch_projects_from_firestore():
    collection_ref = db.collection('projects')  # Replace 'projects' with your actual collection name
    docs = collection_ref.stream()
    project_list = []

    for doc in docs:
        data = doc.to_dict()
        data['id'] = doc.id
        project_list.append(data)
    
    return project_list

# Compute cosine similarity between interests and project info
def compute_cosine_similarity(user_interests, project_domains, project_tools):
    all_texts = [', '.join(user_interests), ', '.join(project_domains + project_tools)]
    vectorizer = TfidfVectorizer().fit_transform(all_texts)
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0][1]

# Recommend projects
def recommend_projects(user_interests):
    user_interests = [i.lower() for i in user_interests]
    projects = fetch_projects_from_firestore()
    recommendations = []

    for project in projects:
        domains = str(project.get('Domain of Work', '')).lower().split(', ')
        tools = str(project.get('Tools used/Using', '')).lower().split(', ')
        score = compute_cosine_similarity(user_interests, domains, tools)

        if score > 0:
            recommendations.append({
                'id': project['id'],
                'title': project.get('Title of the Project', ''),
                'description': project.get('Describe your project/problem statement', ''),
                'domain': project.get('Domain of Work', ''),
                'tools': project.get('Tools used/Using', ''),
                'similarity_score': score
            })

    recommendations = sorted(recommendations, key=lambda x: x['similarity_score'], reverse=True)
    return recommendations

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_interests = data.get("interests", [])
    recommendations = recommend_projects(user_interests)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)