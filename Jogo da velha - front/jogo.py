"""
Jogo da Velha com IA - T1 Inteligencia Artificial PUCRS
Humano (X) vs Maquina aleatoria (O)
A IA classifica o estado do tabuleiro apos cada jogada.

Modelos disponiveis:
  1. KNN           (implementacao manual - knn.ipynb)
  2. Arvore        (sklearn             - tree.ipynb)
  3. MLP           (implementacao manual - MLP.ipynb)
  4. Random Forest (implementacao manual - RandomForest.ipynb)
  5. SVM OvR RBF  (implementacao manual - SVM.ipynb)
"""

import csv
import math
import random
import os
from collections import Counter

import numpy as np

# ─────────────────────────────────────────────
# LABEL MAP (igual ao dos notebooks)
# ─────────────────────────────────────────────
LABEL_MAP = {
    'Tem jogo':          0,
    'Possibilidade_fim': 1,
    'Empate':            2,
    'Fim_O_Vence':       3,
    'Fim_X_Vence':       4,
}
LABEL_NAMES = ['Tem jogo', 'Possibilidade_fim', 'Empate', 'Fim_O_Vence', 'Fim_X_Vence']

NOMES_AMIGAVEIS = {
    'Tem jogo':          'Tem Jogo',
    'Possibilidade_fim': 'Possibilidade de Fim de Jogo',
    'Empate':            'Empate',
    'Fim_O_Vence':       'O Vence',
    'Fim_X_Vence':       'X Vence',
}

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────
# CARREGAR DATASET (mesmo codigo dos notebooks)
# ─────────────────────────────────────────────
def carregar_csv(caminho):
    X, y = [], []
    with open(caminho, 'r') as f:
        for linha in csv.reader(f):
            X.append([float(v) for v in linha[:9]])
            y.append(linha[9])
    return X, y


# ═════════════════════════════════════════════
# MODELO 1 - KNN
# Implementacao identica ao knn.ipynb
# ═════════════════════════════════════════════
class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _distancia(self, a, b):
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))

    def predict_one(self, x):
        distancias = [(self._distancia(x, xi), yi)
                      for xi, yi in zip(self.X_train, self.y_train)]
        distancias.sort(key=lambda t: t[0])
        vizinhos = [c for _, c in distancias[:self.k]]
        return Counter(vizinhos).most_common(1)[0][0]

    def predict(self, X):
        return [self.predict_one(x) for x in X]


def acuracia(y_real, y_pred):
    return sum(1 for r, p in zip(y_real, y_pred) if r == p) / len(y_real)


def encontrar_melhor_k(X_train, y_train, X_test, y_test, ks=(1, 3, 5, 7, 9)):
    melhor_k, melhor_acc = 5, 0
    for k in ks:
        knn = KNN(k=k)
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        acc = acuracia(y_test, pred)
        if acc > melhor_acc:
            melhor_acc, melhor_k = acc, k
    return melhor_k, melhor_acc


# ═════════════════════════════════════════════
# MODELO 2 - ARVORE DE DECISAO
# Implementacao identica ao tree.ipynb (sklearn)
# ═════════════════════════════════════════════
def treinar_arvore(X_train, y_train):
    from sklearn.tree import DecisionTreeClassifier
    import pandas as pd
    colunas = [f'pos_{i}' for i in range(1, 10)]
    Xdf = pd.DataFrame(X_train, columns=colunas)
    modelo = DecisionTreeClassifier(random_state=0, criterion='entropy')
    modelo.fit(Xdf, y_train)
    return modelo


def predizer_arvore(modelo, x):
    import pandas as pd
    colunas = [f'pos_{i}' for i in range(1, 10)]
    Xdf = pd.DataFrame([x], columns=colunas)
    return modelo.predict(Xdf)[0]


# ═════════════════════════════════════════════
# MODELO 3 - MLP
# Implementacao identica ao MLP.ipynb
# Topologia: 9 entradas -> 18 ocultas (sigmoid) -> 5 saidas (softmax)
# Hiperparametros finais do notebook: n_oculta=18, taxa=0.1, epocas=3000, batch=64
# ═════════════════════════════════════════════
def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def _d_sigmoid(x):
    s = _sigmoid(x)
    return s * (1 - s)

def _softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def _one_hot(y, n_classes):
    m = np.zeros((len(y), n_classes))
    m[np.arange(len(y)), y] = 1.0
    return m

def treinar_mlp(X_train_raw, y_train_raw,
                n_oculta=18, taxa=0.1, epocas=3000, batch=64, seed=42):
    X = np.array(X_train_raw)
    y = np.array([LABEL_MAP[c] for c in y_train_raw])
    n_entrada, n_saida = 9, 5

    np.random.seed(seed)
    W1 = np.random.randn(n_entrada, n_oculta) * np.sqrt(2.0 / n_entrada)
    W2 = np.random.randn(n_oculta,  n_saida)  * np.sqrt(2.0 / n_oculta)
    b1 = np.zeros(n_oculta)
    b2 = np.zeros(n_saida)

    for _ in range(epocas):
        idx = np.random.permutation(len(X))
        Xemb, yemb = X[idx], y[idx]
        for ini in range(0, len(Xemb), batch):
            Xb   = Xemb[ini:ini+batch]
            yb   = yemb[ini:ini+batch]
            alvo = _one_hot(yb, n_saida)
            z1 = Xb @ W1 + b1
            a1 = _sigmoid(z1)
            z2 = a1 @ W2 + b2
            a2 = _softmax(z2)
            delta2 = a2 - alvo
            delta1 = (delta2 @ W2.T) * _d_sigmoid(z1)
            W2 -= taxa * (a1.T @ delta2) / len(Xb)
            b2 -= taxa * delta2.mean(axis=0)
            W1 -= taxa * (Xb.T @ delta1) / len(Xb)
            b1 -= taxa * delta1.mean(axis=0)

    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


def predizer_mlp(pesos, x):
    xarr = np.array([x])
    a1   = _sigmoid(xarr @ pesos['W1'] + pesos['b1'])
    a2   = _softmax(a1   @ pesos['W2'] + pesos['b2'])
    return LABEL_NAMES[int(np.argmax(a2))]


# ═════════════════════════════════════════════
# MODELO 4 - RANDOM FOREST
# Implementacao identica ao RandomForest.ipynb
# Hiperparametros finais: n_arvores=100, prof_max=10, n_features=4
# ═════════════════════════════════════════════
def _gini(y_lista):
    n = len(y_lista)
    if n == 0:
        return 0.0
    contagem = Counter(y_lista)
    return 1.0 - sum((c / n) ** 2 for c in contagem.values())


def _melhor_split_rf(X, y, features_idx):
    melhor_ganho = -1
    melhor_feat  = None
    melhor_thr   = None
    gini_pai     = _gini(y)
    n            = len(y)
    for f in features_idx:
        valores  = sorted(set(x[f] for x in X))
        limiares = [(valores[i] + valores[i+1]) / 2 for i in range(len(valores) - 1)]
        for thr in limiares:
            esq_y = [y[i] for i in range(n) if X[i][f] <= thr]
            dir_y = [y[i] for i in range(n) if X[i][f] >  thr]
            if not esq_y or not dir_y:
                continue
            ganho = (gini_pai
                     - (len(esq_y) / n) * _gini(esq_y)
                     - (len(dir_y) / n) * _gini(dir_y))
            if ganho > melhor_ganho:
                melhor_ganho = ganho
                melhor_feat  = f
                melhor_thr   = thr
    return melhor_feat, melhor_thr


def _constroi_arvore_rf(X, y, prof_max, min_amostras, n_features, prof=0):
    if len(y) <= min_amostras or prof >= prof_max or len(set(y)) == 1:
        return Counter(y).most_common(1)[0][0]
    n_total = len(X[0])
    feats   = random.sample(range(n_total), k=min(n_features, n_total))
    feat, thr = _melhor_split_rf(X, y, feats)
    if feat is None:
        return Counter(y).most_common(1)[0][0]
    esq_X = [X[i] for i in range(len(X)) if X[i][feat] <= thr]
    esq_y = [y[i] for i in range(len(y)) if X[i][feat] <= thr]
    dir_X = [X[i] for i in range(len(X)) if X[i][feat] >  thr]
    dir_y = [y[i] for i in range(len(y)) if X[i][feat] >  thr]
    return {
        'feat': feat, 'thr': thr,
        'esq': _constroi_arvore_rf(esq_X, esq_y, prof_max, min_amostras, n_features, prof+1),
        'dir': _constroi_arvore_rf(dir_X, dir_y, prof_max, min_amostras, n_features, prof+1),
    }


def _prediz_arvore_rf(no, x):
    if not isinstance(no, dict):
        return no
    if x[no['feat']] <= no['thr']:
        return _prediz_arvore_rf(no['esq'], x)
    return _prediz_arvore_rf(no['dir'], x)


def _bootstrap_rf(X, y):
    n   = len(X)
    idx = [random.randint(0, n - 1) for _ in range(n)]
    return [X[i] for i in idx], [y[i] for i in idx]


def treinar_random_forest(X_train, y_train,
                          n_arvores=100, prof_max=10,
                          min_amostras=2, n_features=4, seed=42):
    random.seed(seed)
    floresta = []
    for _ in range(n_arvores):
        Xb, yb = _bootstrap_rf(X_train, y_train)
        arvore  = _constroi_arvore_rf(Xb, yb, prof_max, min_amostras, n_features)
        floresta.append(arvore)
    return floresta


def predizer_random_forest(floresta, x):
    votos = [_prediz_arvore_rf(a, x) for a in floresta]
    return Counter(votos).most_common(1)[0][0]


# ═════════════════════════════════════════════
# MODELO 5 - SVM OvR RBF
# Implementacao identica ao SVM.ipynb
# Hiperparametros finais: C=1.0, gamma=0.5, n_por_classe=150, max_passes=10
# ═════════════════════════════════════════════
def _kernel_rbf_matriz(X, gamma=0.5):
    normas = np.sum(X ** 2, axis=1)
    dist2  = normas[:, None] + normas[None, :] - 2.0 * (X @ X.T)
    return np.exp(-gamma * np.maximum(dist2, 0.0))


def _calcula_scores_svm(X_sv, y_sv, alphas, b, X_novo, gamma=0.5):
    sv_mask     = alphas > 1e-5
    normas_sv   = np.sum(X_sv[sv_mask] ** 2, axis=1)
    normas_novo = np.sum(X_novo        ** 2, axis=1)
    dist2       = normas_sv[None, :] + normas_novo[:, None] - 2.0 * (X_novo @ X_sv[sv_mask].T)
    K_novo      = np.exp(-gamma * np.maximum(dist2, 0.0))
    return K_novo @ (alphas[sv_mask] * y_sv[sv_mask]) + b


def _treina_smo(X, y_bin, C=1.0, gamma=0.5, max_passes=10, tol=1e-3):
    n      = len(X)
    alphas = np.zeros(n)
    b      = 0.0
    K      = _kernel_rbf_matriz(X, gamma)
    passes = 0
    while passes < max_passes:
        num_mudancas = 0
        F = (alphas * y_bin) @ K + b
        for i in range(n):
            Ei = F[i] - y_bin[i]
            violacao = ((y_bin[i] * Ei < -tol and alphas[i] < C) or
                        (y_bin[i] * Ei >  tol and alphas[i] > 0))
            if not violacao:
                continue
            j = random.randint(0, n - 1)
            while j == i:
                j = random.randint(0, n - 1)
            Ej     = F[j] - y_bin[j]
            ai_old = alphas[i]
            aj_old = alphas[j]
            if y_bin[i] != y_bin[j]:
                L = max(0.0, alphas[j] - alphas[i])
                H = min(C,   C + alphas[j] - alphas[i])
            else:
                L = max(0.0, alphas[i] + alphas[j] - C)
                H = min(C,   alphas[i] + alphas[j])
            if L >= H:
                continue
            eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
            if eta >= 0:
                continue
            alphas[j] -= y_bin[j] * (Ei - Ej) / eta
            alphas[j]  = float(np.clip(alphas[j], L, H))
            if abs(alphas[j] - aj_old) < 1e-5:
                continue
            alphas[i] += y_bin[i] * y_bin[j] * (aj_old - alphas[j])
            b1v = (b - Ei
                   - y_bin[i] * (alphas[i] - ai_old) * K[i, i]
                   - y_bin[j] * (alphas[j] - aj_old) * K[i, j])
            b2v = (b - Ej
                   - y_bin[i] * (alphas[i] - ai_old) * K[i, j]
                   - y_bin[j] * (alphas[j] - aj_old) * K[j, j])
            if   0 < alphas[i] < C: b = b1v
            elif 0 < alphas[j] < C: b = b2v
            else:                   b = (b1v + b2v) / 2.0
            F = (alphas * y_bin) @ K + b
            num_mudancas += 1
        passes = passes + 1 if num_mudancas == 0 else 0
    return alphas, b


def _subconjunto_svm(X_raw, y_raw, n_por_classe=150, seed=42):
    X_arr = np.array(X_raw)
    y_num = np.array([LABEL_MAP[c] for c in y_raw])
    rng   = np.random.default_rng(seed)
    indices = []
    for c in range(5):
        idx_c = np.where(y_num == c)[0]
        rng.shuffle(idx_c)
        indices.extend(idx_c[:n_por_classe].tolist())
    return X_arr[indices], y_num[indices]


def treinar_svm(X_train, y_train,
                C=1.0, gamma=0.5, max_passes=10, n_por_classe=150, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    X_sub, y_sub = _subconjunto_svm(X_train, y_train, n_por_classe, seed)
    classificadores = []
    for c in range(5):
        y_bin = np.where(y_sub == c, 1.0, -1.0)
        alphas, b = _treina_smo(X_sub, y_bin, C=C, gamma=gamma, max_passes=max_passes)
        classificadores.append((X_sub, y_bin, alphas, b))
    return classificadores, gamma


def predizer_svm(classificadores, gamma, x):
    x_arr  = np.array([x])
    scores = [_calcula_scores_svm(Xs, yb, al, b, x_arr, gamma)[0]
              for Xs, yb, al, b in classificadores]
    return LABEL_NAMES[int(np.argmax(scores))]


# ─────────────────────────────────────────────
# TABULEIRO -> VETOR (mesmo padrao dos notebooks)
# X=1, O=-1, vazio=0
# ─────────────────────────────────────────────
def tabuleiro_para_vetor(tab):
    mapa = {'X': 1.0, 'O': -1.0, ' ': 0.0}
    return [mapa[cel] for linha in tab for cel in linha]


# ─────────────────────────────────────────────
# ESTADO REAL DO JOGO (regras normais)
# ─────────────────────────────────────────────
def estado_real(tab):
    linhas    = tab
    colunas   = [[tab[i][j] for i in range(3)] for j in range(3)]
    diagonais = [
        [tab[0][0], tab[1][1], tab[2][2]],
        [tab[0][2], tab[1][1], tab[2][0]],
    ]
    for grupo in linhas + colunas + diagonais:
        if grupo == ['X', 'X', 'X']:
            return 'Fim_X_Vence'
        if grupo == ['O', 'O', 'O']:
            return 'Fim_O_Vence'
    if all(tab[i][j] != ' ' for i in range(3) for j in range(3)):
        return 'Empate'
    for grupo in linhas + colunas + diagonais:
        if grupo.count('X') == 2 and grupo.count(' ') == 1:
            return 'Possibilidade_fim'
        if grupo.count('O') == 2 and grupo.count(' ') == 1:
            return 'Possibilidade_fim'
    return 'Tem jogo'


def eh_fim_de_jogo(estado):
    return estado in ('Fim_X_Vence', 'Fim_O_Vence', 'Empate')


# ─────────────────────────────────────────────
# EXIBICAO
# ─────────────────────────────────────────────
def limpar():
    os.system('cls' if os.name == 'nt' else 'clear')


def mostrar_tabuleiro(tab):
    print()
    for i in range(3):
        print('  ' + ' | '.join(f' {cel} ' for cel in tab[i]))
        if i < 2:
            print('  ---+-----+---')
    print()


def mostrar_posicoes():
    print('  Posicoes disponiveis:')
    print('   1 | 2 | 3')
    print('   4 | 5 | 6')
    print('   7 | 8 | 9')
    print()


def barra(titulo, largura=55):
    print('=' * largura)
    print(f'  {titulo}')
    print('=' * largura)


def mostrar_score(acertos, erros, total):
    acc = (acertos / total * 100) if total > 0 else 0
    print(f'  IA - Acertos: {acertos} | Erros: {erros} | '
          f'Total: {total} | Acuracia: {acc:.1f}%')


# ─────────────────────────────────────────────
# PREDICAO UNIFICADA
# ─────────────────────────────────────────────
def predizer(modelo, tipo, vetor):
    if tipo == 'knn':
        return modelo.predict_one(vetor)
    elif tipo == 'arvore':
        return predizer_arvore(modelo, vetor)
    elif tipo == 'mlp':
        return predizer_mlp(modelo, vetor)
    elif tipo == 'rf':
        return predizer_random_forest(modelo, vetor)
    elif tipo == 'svm':
        clf, gamma = modelo
        return predizer_svm(clf, gamma, vetor)


# ─────────────────────────────────────────────
# JOGADA DO USUARIO
# ─────────────────────────────────────────────
def jogada_usuario(tab):
    while True:
        mostrar_posicoes()
        entrada = input('  Sua jogada (1-9): ').strip()
        if not entrada.isdigit():
            print('  >> Digite um numero de 1 a 9.')
            continue
        pos = int(entrada)
        if pos < 1 or pos > 9:
            print('  >> Numero invalido. Digite de 1 a 9.')
            continue
        i, j = (pos - 1) // 3, (pos - 1) % 3
        if tab[i][j] != ' ':
            print('  >> Posicao ocupada! Tente outra.')
            continue
        tab[i][j] = 'X'
        return pos


# ─────────────────────────────────────────────
# JOGADA DA MAQUINA (aleatoria)
# ─────────────────────────────────────────────
def jogada_maquina(tab):
    vazios = [(i, j) for i in range(3) for j in range(3) if tab[i][j] == ' ']
    if vazios:
        i, j = random.choice(vazios)
        tab[i][j] = 'O'
        return i * 3 + j + 1
    return None


# ─────────────────────────────────────────────
# AVALIAR JOGADA
# ─────────────────────────────────────────────
def avaliar_jogada(modelo, tipo, tab, acertos, erros):
    vetor   = tabuleiro_para_vetor(tab)
    pred    = predizer(modelo, tipo, vetor)
    real    = estado_real(tab)
    acertou = (pred == real)

    if acertou:
        acertos += 1
        icone = '[OK]'
    else:
        erros += 1
        icone = '[ERRO]'

    total = acertos + erros
    acc   = (acertos / total * 100) if total > 0 else 0

    print(f'  IA previu   : {NOMES_AMIGAVEIS.get(pred, pred)}')
    print(f'  Estado real : {NOMES_AMIGAVEIS.get(real, real)}')
    print(f'  Resultado   : {icone}')
    print(f'  Score       : {acertos} acertos, {erros} erros | Acuracia: {acc:.1f}%')
    print()

    return pred, real, acertos, erros


# ─────────────────────────────────────────────
# RESULTADO FINAL
# ─────────────────────────────────────────────
def resultado_final(acertos, erros):
    total = acertos + erros
    acc   = (acertos / total * 100) if total > 0 else 0
    print('─' * 55)
    print('  RESULTADO FINAL DA IA')
    print('─' * 55)
    print(f'  Total de previsoes : {total}')
    print(f'  Acertos            : {acertos}')
    print(f'  Erros              : {erros}')
    print(f'  Acuracia           : {acc:.2f}%')
    print('─' * 55)
    input('\n  [Enter para voltar ao menu]')


# ─────────────────────────────────────────────
# LOOP PRINCIPAL DO JOGO
# ─────────────────────────────────────────────
def jogar(modelo, tipo, nome_modelo):
    tab     = [[' '] * 3 for _ in range(3)]
    acertos = 0
    erros   = 0
    turno   = 1

    while True:
        limpar()
        barra(f'Jogo da Velha - Modelo: {nome_modelo}')
        mostrar_tabuleiro(tab)
        mostrar_score(acertos, erros, acertos + erros)
        print()

        # ── TURNO DO USUARIO ──────────────────────
        print('  === Sua vez (X) ===')
        jogada_usuario(tab)
        mostrar_tabuleiro(tab)
        print(f'  [Turno {turno}] Apos sua jogada:')
        pred, real, acertos, erros = avaliar_jogada(modelo, tipo, tab, acertos, erros)

        fim_real = eh_fim_de_jogo(real)
        fim_pred = eh_fim_de_jogo(pred)

        if fim_real and not fim_pred:
            # IA errou o fim -> encerrar imediatamente (regra do enunciado)
            barra('PARTIDA ENCERRADA - IA nao detectou o fim')
            print(f'  Fim real : {NOMES_AMIGAVEIS[real]}')
            print(f'  IA disse : {NOMES_AMIGAVEIS.get(pred, pred)}')
            print()
            resultado_final(acertos, erros)
            return

        if fim_real and fim_pred:
            barra('PARTIDA ENCERRADA')
            print(f'  {NOMES_AMIGAVEIS[real]}  (IA detectou corretamente!)')
            print()
            resultado_final(acertos, erros)
            return

        if not fim_real and fim_pred:
            # IA detectou fim incorretamente -> continuar (regra do enunciado)
            print('  >> IA acha que acabou, mas o jogo continua...')
            print()

        # ── TURNO DA MAQUINA ──────────────────────
        pos = jogada_maquina(tab)
        if pos is None:
            print('  [Sem casas livres!]')
            resultado_final(acertos, erros)
            return

        mostrar_tabuleiro(tab)
        print(f'  [Turno {turno}] Maquina jogou na posicao {pos}:')
        pred, real, acertos, erros = avaliar_jogada(modelo, tipo, tab, acertos, erros)

        fim_real = eh_fim_de_jogo(real)
        fim_pred = eh_fim_de_jogo(pred)

        if fim_real and not fim_pred:
            barra('PARTIDA ENCERRADA - IA nao detectou o fim')
            print(f'  Fim real : {NOMES_AMIGAVEIS[real]}')
            print(f'  IA disse : {NOMES_AMIGAVEIS.get(pred, pred)}')
            print()
            resultado_final(acertos, erros)
            return

        if fim_real and fim_pred:
            barra('PARTIDA ENCERRADA')
            print(f'  {NOMES_AMIGAVEIS[real]}  (IA detectou corretamente!)')
            print()
            resultado_final(acertos, erros)
            return

        if not fim_real and fim_pred:
            print('  >> IA acha que acabou, mas o jogo continua...')
            print()

        input('  [Enter para continuar...]')
        turno += 1


# ─────────────────────────────────────────────
# MENU DE SELECAO DE MODELO
# ─────────────────────────────────────────────
def menu_modelos():
    print()
    barra('Escolha o modelo de IA')
    print('  1. KNN           ')
    print('  2. Arvore        ')
    print('  3. MLP           ')
    print('  4. Random Forest ')
    print('  5. SVM           ')
    print()
    escolha = input('  Digite 1-5 [Enter = KNN]: ').strip()
    if escolha not in ('1', '2', '3', '4', '5'):
        escolha = '1'
    return escolha


# ─────────────────────────────────────────────
# INICIALIZAR MODELO ESCOLHIDO
# ─────────────────────────────────────────────
def inicializar_modelo(escolha):
    caminho_treino = os.path.join(DATASET_DIR, 'dataset_treino.csv')
    caminho_teste  = os.path.join(DATASET_DIR, 'dataset_testes.csv')

    print()
    print('  Carregando datasets...')
    X_train, y_train = carregar_csv(caminho_treino)
    X_test,  y_test  = carregar_csv(caminho_teste)
    print(f'  Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras')
    print()

    if escolha == '1':
        print('  [KNN] Buscando melhor K (testando 1, 3, 5, 7, 9)...')
        melhor_k, acc = encontrar_melhor_k(X_train, y_train, X_test, y_test)
        print(f'  Melhor K = {melhor_k}  (acuracia no teste: {acc*100:.1f}%)')
        knn = KNN(k=melhor_k)
        knn.fit(X_train, y_train)
        return knn, 'knn', f'KNN (K={melhor_k})'

    elif escolha == '2':
        print('  [Arvore] Treinando arvore de decisao...')
        arvore = treinar_arvore(X_train, y_train)
        pred   = [predizer_arvore(arvore, x) for x in X_test]
        acc    = acuracia(y_test, pred)
        print(f'  Arvore treinada  (acuracia no teste: {acc*100:.1f}%)')
        return arvore, 'arvore', 'Arvore de Decisao'

    elif escolha == '3':
        print('  [MLP] Treinando rede neural (9->18->5, 3000 epocas)...')
        print('  Aguarde alguns segundos...')
        pesos = treinar_mlp(X_train, y_train)
        pred  = [predizer_mlp(pesos, x) for x in X_test]
        acc   = acuracia(y_test, pred)
        print(f'  MLP treinada     (acuracia no teste: {acc*100:.1f}%)')
        return pesos, 'mlp', 'MLP (9->18->5)'

    elif escolha == '4':
        print('  [Random Forest] Treinando 100 arvores...')
        print('  Aguarde alguns segundos...')
        floresta = treinar_random_forest(X_train, y_train)
        pred     = [predizer_random_forest(floresta, x) for x in X_test]
        acc      = acuracia(y_test, pred)
        print(f'  Random Forest treinado  (acuracia no teste: {acc*100:.1f}%)')
        return floresta, 'rf', 'Random Forest (100 arvores)'

    elif escolha == '5':
        print('  [SVM] Treinando 5 classificadores OvR com kernel RBF...')
        print('  Aguarde alguns segundos...')
        clf, gamma = treinar_svm(X_train, y_train)
        pred = [predizer_svm(clf, gamma, x) for x in X_test]
        acc  = acuracia(y_test, pred)
        print(f'  SVM treinado     (acuracia no teste: {acc*100:.1f}%)')
        return (clf, gamma), 'svm', 'SVM OvR RBF'


# ─────────────────────────────────────────────
# MENU PRINCIPAL
# ─────────────────────────────────────────────
def main():
    modelo_atual = None
    tipo_atual   = None
    nome_atual   = None

    while True:
        limpar()
        barra('JOGO DA VELHA com IA - T1 IA PUCRS')
        print('  Humano (X) vs Maquina Aleatoria (O)')
        print()
        if modelo_atual:
            print(f'  Modelo carregado: {nome_atual}')
        else:
            print('  Nenhum modelo carregado ainda.')
        print()
        print('  1. Selecionar modelo e jogar')
        print('  2. Jogar com modelo atual')
        print('  0. Sair')
        print()
        op = input('  Opcao: ').strip()

        if op == '0':
            print('\n  Ate mais!\n')
            break

        elif op == '1':
            escolha = menu_modelos()
            limpar()
            barra('Carregando modelo...')
            try:
                modelo_atual, tipo_atual, nome_atual = inicializar_modelo(escolha)
                print(f'\n  Modelo "{nome_atual}" pronto!')
                input('\n  [Enter para comecar o jogo]')
                jogar(modelo_atual, tipo_atual, nome_atual)
            except FileNotFoundError as e:
                print(f'\n  ERRO: {e}')
                print('  Certifique-se de que dataset_treino.csv e dataset_testes.csv')
                print('  estao na mesma pasta que jogo.py')
                input('\n  [Enter para continuar]')

        elif op == '2':
            if modelo_atual is None:
                print('\n  Nenhum modelo carregado. Selecione primeiro (opcao 1).')
                input('  [Enter para continuar]')
            else:
                jogar(modelo_atual, tipo_atual, nome_atual)

        else:
            print('\n  Opcao invalida.')
            input('  [Enter para continuar]')


if __name__ == '__main__':
    main()
