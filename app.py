from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import random
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import re

nltk.download('stopwords')
nltk.download('rslp')

app = Flask(__name__)
CORS(app)  # Ativando CORS

# Carregando dados
with open("perguntas.json", "r") as file:
    perguntas_data = json.load(file)

# Carregando respostas do arquivo JSON
with open("respostas.json", "r") as file:
    respostas = json.load(file)

perguntas = []
intencoes = []

# Organizando perguntas e intenções
for intencao, lista_perguntas in perguntas_data.items():
    perguntas.extend(lista_perguntas)
    intencoes.extend([intencao] * len(lista_perguntas))

# Definindo stopwords e stemmer em português
stop_words = set(stopwords.words('portuguese'))
stemmer = RSLPStemmer()

# Função para pré-processamento
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Pré-processando perguntas
perguntas_preprocessadas = [preprocess_text(pergunta) for pergunta in perguntas]

# Configurando o Vectorizer
vectorizer = TfidfVectorizer(lowercase=True, max_df=0.85, min_df=2, ngram_range=(1, 2))
X = vectorizer.fit_transform(perguntas_preprocessadas)

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X, intencoes)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    pergunta_usuario = preprocess_text(data.get("message"))
    pergunta_vetor = vectorizer.transform([pergunta_usuario])

    similaridades = cosine_similarity(pergunta_vetor, X)
    max_similaridade = similaridades.max()

    if max_similaridade < 0.5:
        return jsonify({"response": "Desculpe, não consegui entender sua pergunta. Pode reformular?"})

    intencao = modelo.predict(pergunta_vetor)[0]
    resposta = random.choice(respostas[intencao])
    return jsonify({"response": resposta})

if __name__ == '__main__':
    app.run(debug=True)
