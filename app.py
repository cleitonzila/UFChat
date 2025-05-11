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
        "O Restaurante Universitário tem como principal objetivo proporcionar refeições de alta qualidade para estudantes da UFC. Você pode consultar o site do RU aqui para saber mais: \nhttps://www.ufc.br/restaurante",
        "Veja aqui no site do restaurante universitário:\nhttps://www.ufc.br/restaurante",
        "A UFC do campus Pici possui dois RU’s. Veja as localizações aqui:\nhttps://www.ufc.br/restaurante"
    ],
    "boas_vindas": [
        "Olá! Como posso te ajudar hoje?",
        "Posso te ajudar com alguma dúvida sobre a UFC?",
        "Precisa de ajuda com algo?",
        "Olá :). Posso te ajudar em algo?"
    ],
    "xerox": [
        "Você pode tirar xerox nos blocos 806, 902 e 924 por R$ 0,50 a folha.",
        "Para imprimir seu material de estudo ou de trabalho, entre outras coisas, temos a gráfica digital Sílvio Cópias, no Campus do Pici, que abre às 8:30 da manhã. \nhttps://www.instagram.com/silviocopias?igsh=MTBlenU2Y3Q2OWlqYg=="
    ],
    "xerox_preco": [
        "Para saber o preço dos serviços, entre em contato com a gráfica pelo whatsapp: (85) 98760 2614"
    ],
    "xerox_especificidades": [
        "Entre em contato com o vendedor para saber mais sobre os produtos disponíveis: (85) 98760 2614"
    ],
    "xerox_localizacao": [
        "Você pode tirar xerox nos blocos 806, 902 e 924 do Campus do Pici por R$ 0,50 a folha."
    ],
    "matricula": [
        "Você pode descobrir acessando o SIGAA, neste link:\nhttps://si3.ufc.br/sigaa/verTelaLogin.do",
        "Acesse o SIGAA no seguinte link:\nhttps://si3.ufc.br/sigaa/verTelaLogin.do"
    ],
    "saudacao": [
        "Olá! Como posso te ajudar hoje? 😊",
        "Oi! Tudo bem? Precisa de alguma informação sobre a universidade?",
        "Olá! Sou o assistente da UFC. Em que posso te ajudar?"
    ],
    "perguntas_blocos": [
        "A UFC é gigante, para não acabar se perdendo, dê uma olhada nos mapas dos campi clicando no link! https://umap.openstreetmap.fr/en/map/universidade-federal-do-ceara-ufc-ce_88070#16/-3.7469/-38.5764"
    ],
    "perguntas_matricula": [
        "Parabéns pelo resultado! Vou te ajudar a realizar sua matrícula"
    ],
    "perguntas_cotas": [
        "Aqui estão as etapas de matrículas para cotistas: https://sisu.ufc.br/pt/etapas-de-matricula-2025/"
    ],
    "perguntas_autodeclaracao": [
        "Se você entrou na UFC por meio de cotas raciais, é necessário passar por uma avaliação de características físicas. Saiba mais detalhes sobre o vídeo de autodeclaração aqui: https://sisu.ufc.br/pt/duvidas-frequentes-2025/duvidas-frequentes-lei-de-cotas-video-de-autodeclaracao-e-heteroidentificacao-2025-faq/"
    ],
    "perguntas_heteroidentificacao": [
        "Se você entrou na UFC por meio de cotas raciais, é necessário passar por uma avaliação de características físicas. Saiba mais detalhes sobre o procedimento de heteroidentificação aqui: https://sisu.ufc.br/pt/duvidas-frequentes-2025/duvidas-frequentes-lei-de-cotas-video-de-autodeclaracao-e-heteroidentificacao-2025-faq/"
    ],
    "perguntas_lista_espera": [
        "Caso você não tenha entrado pela chamada regular, você pode entrar pela lista de espera! Aqui estão as informações sobre como fazer isso: https://sisu.ufc.br/pt/lista-de-espera-2025-2/"
    ],
    "perguntas_documentacao": [
        "Alguns documentos são necessários para fazer a matrícula. Saiba mais clicando no link: https://sisu.ufc.br/pt/documentacao-basica-para-matricula/",
        "Caso você seja cotista, cheque a documentação específica para a entrada com cotas."
    ],
    "perguntas_datas": [
        "O cronograma com as datas já deve estar disponível no site: https://sisu.ufc.br/pt/"
    ],
    "pergunta_menor_de_idade": [
        "Não é necessário acompanhamento de responsável, o aluno menor de idade pode ir sozinho"
    ]
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
