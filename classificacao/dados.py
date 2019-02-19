import csv

def carregar_acessos():
    X = []
    Y = []

    arquivo = open('acesso.csv', 'r')
    leitor = csv.reader(arquivo)
    
    #Pular a primeira linha que possui o cabeçalho
    next(leitor)

    for home,como_funciona,contato,comprou in leitor:
        X.append([int(home), 
            int(como_funciona), 
            int(contato)])
        Y.append(int(comprou))
        
    return X, Y


def carregar_buscas():
    X = []
    Y = []

    arquivo = open('busca.csv', 'r')
    leitor = csv.reader(arquivo)

    next(leitor)

    for home,busca,logado,comprou in leitor:
        dado = [
            int(home), 
            busca, 
            int(logado)]
        X.append(dado)
        Y.append(int(comprou))

    return X, Y