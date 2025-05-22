import nltk
nltk.download('stopwords')

import spacy
import os
try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
import spacy
import os
import re
import string
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

# Setup
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)
CORS(app)

# Skills list and synonyms
it_skills = [
    # Programming Languages
    'python', 'java', 'c', 'c++','cpp', 'c#', 'javascript', 'typescript', 'ruby', 'swift', 'kotlin', 'go', 'rust', 'scala', 'php', 'perl',

    # Web Development
    'html','html5', 'css', 'css3','sass', 'less', 'bootstrap', 'react.js','react','js', 'angular', 'vue', 'jquery', 'node.js', 'express.js', 'next.js', 'nuxt.js',
    'webpack', 'babel', 'grunt', 'gulp',

    # Mobile Development
    'android', 'ios', 'react native', 'flutter', 'xamarin', 'cordova', 'ionic',

    # Databases & Data Management
    'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'elasticsearch', 'sqlite', 'mariadb', 'oracle', 'dynamodb', 'firebase',

    # Machine Learning & Data Science
    'tensorflow', 'keras', 'pytorch', 'scikit-learn', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'opencv', 'nltk', 'spacy', 'xgboost',
    'lightgbm', 'catboost', 'mlflow', 'fastai', 'statsmodels',

    # Cloud Computing
    'amazon web sevices','aws', 'azure', 'google cloud', 'gcp', 'ec2', 's3', 'lambda', 'cloudformation', 'cloudfront', 'iam', 'vpc',
    'docker', 'kubernetes', 'helm', 'terraform', 'ansible', 'openshift',

    # DevOps & CI/CD
    'jenkins', 'gitlab ci', 'circleci', 'travis ci', 'teamcity', 'bamboo', 'prometheus', 'grafana', 'elasticsearch', 'logstash', 'kibana',
    'nagios', 'puppet', 'chef',

    # Cybersecurity
    'penetration testing', 'ethical hacking', 'network security', 'firewall', 'intrusion detection', 'vulnerability assessment',
    'siem', 'cryptography', 'cissp', 'ceh', 'compTIA security+', 'owasp', 'malware analysis',

    # AI & NLP
    'natural language processing', 'computer vision', 'deep learning', 'reinforcement learning', 'transformers', 'bert', 'gpt', 'nlp',

    # Big Data & Streaming
    'hadoop', 'spark', 'kafka', 'apache kafka', 'flink', 'storm', 'hive', 'pig', 'zeppelin',

    # Testing & Quality Assurance
    'selenium', 'junit', 'pytest', 'cucumber', 'mocha', 'chai', 'jasmine', 'postman', 'rest assured',

    # Tools & Platforms
    'git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence', 'slack', 'docker-compose', 'vagrant',

    # Others
    'microservices', 'graphql', 'rest api', 'soap', 'oauth', 'jwt', 'rabbitmq', 'mqtt', 'websockets',
    'agile', 'scrum', 'kanban', 'tdd', 'bdd', 'ci/cd', 'mvc', 'mvvm',
]


skill_synonyms = {
    "html5": "html", "html": "html",
    "css3": "css", "css": "css",
    "js": "javascript", "javascript": "javascript",
    "react": "react.js", "react.js": "react.js",
    "node": "node.js", "node.js": "node.js",
    "cpp": "c++", "c++": "c++"
}

# Utility Functions
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if token.text not in stop_words and token.text not in string.punctuation]
    return " ".join(tokens)

def extract_skills_section(text):
    skills_keywords = ['skills', 'technical skills', 'key skills']
    lines = text.lower().split('\n')
    skills_section = ""
    capture = False
    for line in lines:
        if any(kw in line for kw in skills_keywords):
            capture = True
            continue
        if capture:
            if line.strip() == "" or len(line.strip()) < 3:
                continue
            if any(heading in line for heading in ['experience','education', 'projects','summary','objective','certifications']):
                break
            skills_section += line + " "
    return skills_section.strip()

def extract_skills(text, skills_list):
    text = text.lower()
    found_skills = set()
    for skill in skills_list:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text):
            normalized = skill_synonyms.get(skill.lower(), skill.lower())
            found_skills.add(normalized)
    return sorted(found_skills)

def calculate_similarity(resume_text, job_desc_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_desc_text])
    return cosine_similarity(vectors)[0][1]

def calculate_skill_score(resume_skills, jd_skills):
    matched = set(resume_skills).intersection(set(jd_skills))
    if len(jd_skills) == 0:
        return 0.0, [], jd_skills
    return len(matched) / len(jd_skills), list(matched), list(set(jd_skills) - set(resume_skills))

# 🎯 Flask Endpoint
@app.route('/match', methods=['POST'])
def match_resume():
    file = request.files['resume']
    job_desc = request.form['jd']

    resume_raw_text = extract_text_from_pdf(file)
    skills_section_text = extract_skills_section(resume_raw_text)

    cleaned_resume = preprocess(resume_raw_text)
    cleaned_jd = preprocess(job_desc)

    resume_skills = extract_skills(skills_section_text, it_skills)
    jd_skills = extract_skills(job_desc, it_skills)

    skill_score, matched_skills, missing_skills = calculate_skill_score(resume_skills, jd_skills)

    return jsonify({
        "resume_skills": resume_skills,
        "jd_skills": jd_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "skill_score": round(skill_score, 4)
    })

# 🔁 Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
