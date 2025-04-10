from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load your CSV file
projects_df = pd.read_csv("details.csv", encoding='latin1')  # or 'windows-1252'

# Similarity computation logic
def compute_cosine_similarity(user_interests, project_domains, project_tools):
    all_texts = [', '.join(user_interests), ', '.join(project_domains + project_tools)]
    vectorizer = TfidfVectorizer().fit_transform(all_texts)
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0][1]

def recommend_projects(user_interests):
    recommendations = []
    for _, row in projects_df.iterrows():
        domains = str(row['Domain of Work']).lower().split(', ')
        tools = str(row['Tools used/Using']).lower().split(', ')
        score = compute_cosine_similarity(user_interests, domains, tools)
        if score > 0:
            recommendations.append({
                'title': row['Title of the Project'],
                'description': row['Describe your project/problem statement'],
                'domain': row['Domain of Work'],
                'tools': row['Tools used/Using'],
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
