import os
from collections import Counter

# Caminhos (relativos ao arquivo atual)
BASE_DIR = os.path.dirname(__file__)
INPUT_PATH = os.path.join(BASE_DIR, "tic-tac-toe.data")
OUTPUT_PATH = os.path.join(BASE_DIR, "dataset_tratado.csv")


def converter_valor(v):
    if v == 'x':
        return 1
    elif v == 'o':
        return -1
    else:
        return 0


def classificar(tab):
    linhas = [
        [0,1,2],[3,4,5],[6,7,8],
        [0,3,6],[1,4,7],[2,5,8],
        [0,4,8],[2,4,6]
    ]

    # 1. Vitória
    for linha in linhas:
        soma = tab[linha[0]] + tab[linha[1]] + tab[linha[2]]
        if soma == 3:
            return "X_vence"
        if soma == -3:
            return "O_vence"

    # 2. Empate
    if 0 not in tab:
        return "Empate"

    # 3. Possibilidade de fim
    for linha in linhas:
        valores = [tab[i] for i in linha]
        if valores.count(1) == 2 and valores.count(0) == 1:
            return "Possibilidade_fim"
        if valores.count(-1) == 2 and valores.count(0) == 1:
            return "Possibilidade_fim"

    # 4. Tem jogo
    return "Tem_jogo"


def transformar_dataset():
    novo_dataset = []

    with open(INPUT_PATH, "r") as f:
        for linha in f:
            partes = linha.strip().split(",")

            # pegar só o tabuleiro (ignorar última coluna)
            tabuleiro = partes[:9]

            # converter para números
            tab_num = [converter_valor(v) for v in tabuleiro]

            # classificar
            classe = classificar(tab_num)

            novo_dataset.append(tab_num + [classe])

    return novo_dataset


def salvar_dataset(dataset):
    with open(OUTPUT_PATH, "w") as f:
        for linha in dataset:
            f.write(",".join(map(str, linha)) + "\n")


def main():
    dataset = transformar_dataset()

    # salvar
    salvar_dataset(dataset)

    # mostrar algumas linhas
    print("Exemplos:")
    for linha in dataset[:5]:
        print(linha)

    # contagem de classes
    classes = [linha[-1] for linha in dataset]
    print("\nDistribuição das classes:")
    print(Counter(classes))


if __name__ == "__main__":
    main()