#!-*- coding: utf8 -*-
import numpy as np
import nltk
from sklearn.model_selection import cross_val_score
from collections import Counter
import pandas as pd 

# ======= essa linha de download precisa ser executada caso seja a primeira vez na máquina para baixar a biblioteca =============
# nltk.download('stopwords')
# nltk.download('rslp')  ## para lidar com os sufixos das palavras. Exemplo: não colocar, carreira e carreiras na biblioteca.
# nltk.download("punkt") ## para lidar com as pontuações

texto1 = "Se eu comprar cinco anos antecipados, eu ganho algum desconto?"
texto2 = "O exercício quinze do curso de Java 1 está com a respota errada. Pode conferir pf"
texto3 = "Existe algum curso para cuidar do marketing da minha empresa?"

classificacoes = pd.read_csv('emails.csv', encoding='utf-8')
textosPuros = classificacoes['email']
frases = textosPuros.str.lower()
textosQuebrados = [nltk.tokenize.word_tokenize(frase) for frase in frases]
stopwords = nltk.corpus.stopwords.words("portuguese")
stemmer = nltk.stem.RSLPStemmer()


dicionario = set()
for lista in textosQuebrados:
    validas = [stemmer.stem(palavra) for palavra in lista if palavra not in stopwords and len(palavra) > 2]
    dicionario.update(validas)
 
totalDePalavras = len(dicionario)
print("Total de palavras no dicionário:", totalDePalavras)

# junta cada palavra a um numero chave incremental
tuplas = zip(dicionario, range(totalDePalavras))
tradutor = {palavra:indice for palavra,indice in tuplas}

def vetorizar_texto(texto, tradutor):
    vetor = [0] * len(tradutor)

    for palavra in texto:
        if len(palavra) > 0:
            raiz = stemmer.stem(palavra)
            if raiz in tradutor:
                posicao = tradutor[raiz]
                vetor[posicao] += 1

    return vetor

vetoresDeTexto = [vetorizar_texto(texto, tradutor) for texto in textosQuebrados]
marcas = classificacoes['classificacao']

X = np.array(vetoresDeTexto)
Y = np.array(marcas)

porcentagem_de_treino = 0.8

tamanho_do_treino = int(porcentagem_de_treino * len(Y))
tamanho_de_validacao = len(Y) - tamanho_do_treino

treino_dados = X[0:tamanho_do_treino]
treino_marcacoes = Y[0:tamanho_do_treino]

validacao_dados = X[tamanho_do_treino:]
validacao_marcacoes = Y[tamanho_do_treino:]



def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes):
    k = 10
    scores = cross_val_score(modelo, treino_dados, treino_marcacoes, cv= k)
    taxa_de_acerto = np.mean(scores)

    print("Taxa de acerto do {0}: {1}".format(nome, taxa_de_acerto))
    return taxa_de_acerto


def teste_real(modelo, validacao_dados, validacao_marcacoes):
    resultado = modelo.predict(validacao_dados)
    acertos = (resultado == validacao_marcacoes)

    total_de_acertos = sum(acertos)
    total_de_elementos = len(validacao_marcacoes)
    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

    print("Taxa de acerto do vencedor no mundo real: {0}".format(taxa_de_acerto))



resultados = {}

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state = 0))
resultadoOneVsRest = fit_and_predict("OneVsRest", modeloOneVsRest, treino_dados, treino_marcacoes)
resultados[resultadoOneVsRest] = modeloOneVsRest

from sklearn.multiclass import OneVsOneClassifier
modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state=0))
resultadoOneVsOne = fit_and_predict("OneVsOne", modeloOneVsOne, treino_dados, treino_marcacoes)
resultados[resultadoOneVsOne] = modeloOneVsOne

from sklearn.naive_bayes import MultinomialNB
modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes)
resultados[resultadoMultinomial] = modeloMultinomial 

from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes)
resultados[resultadoAdaBoost] = modeloAdaBoost

maximo = max(resultados)
vencedor = resultados[maximo]
vencedor.fit(treino_dados, treino_marcacoes)

teste_real(vencedor, validacao_dados, validacao_marcacoes)

acerto_base = max(Counter(validacao_marcacoes).values())
taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
print("Taxa de acerto base: %f" % taxa_de_acerto_base)
print("Total de testes: %d" % len(validacao_dados))