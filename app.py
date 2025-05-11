from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import random
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # Ativando CORS

# Carregando dados
with open("perguntas.json", "r") as file:
    perguntas_data = json.load(file)

perguntas = []
intencoes = []

# Organizando perguntas e intenções
for intencao, lista_perguntas in perguntas_data.items():
    perguntas.extend(lista_perguntas)
    intencoes.extend([intencao] * len(lista_perguntas))

respostas = {
    "RU": [
        "O Restaurante Universitário tem como principal objetivo proporcionar refeições de alta qualidade para estudantes da UFC. Voce pode consultar o site do RU aqui para saber mais: \nhttps://www.ufc.br/restaurante",
        "Veja aqui no site do restaurante universitário:\n https://www.ufc.br/restaurante",
        "A UFC do campo Pici possui dois RU’s. Veja as localização aqui:\nhttps://www.ufc.br/restaurante"
    ],
    "boas_vindas": [
        "Olá! Como posso te ajudar hoje?",
        "Posso te ajudar com alguma dúvida sobre a UFC?",
        "Precisa de ajuda com algo?", "Ola :). Posso te ajudar em algo?"
    ],
    "xerox": ["Você pode tirar xerox nos blocos 806, 902 e 924 por R$ 0,50 a folha."],
    "matricula": [
        "Você pode descobrir acessando o SIGAA, neste link:\nhttps://si3.ufc.br/sigaa/verTelaLogin.do ",
        "Acesse o SIGAA no seguinte link:\nhttps://si3.ufc.br/sigaa/verTelaLogin.do"
    ],
}

vectorizer = TfidfVectorizer(lowercase=True)
X = vectorizer.fit_transform(perguntas)
modelo = LogisticRegression(max_iter=200)
modelo.fit(X, intencoes)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    pergunta_usuario = data.get("message").lower()
    pergunta_vetor = vectorizer.transform([pergunta_usuario])

    similaridades = cosine_similarity(pergunta_vetor, X)
    max_similaridade = similaridades.max()

    if max_similaridade < 0.3:
        return jsonify({"response": "Desculpe, não consegui entender sua pergunta. Pode reformular?"})

    intencao = modelo.predict(pergunta_vetor)[0]
    resposta = random.choice(respostas[intencao])
    return jsonify({"response": resposta})

if __name__ == '__main__':
    app.run(debug=True)
