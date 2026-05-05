# Jogo da Velha com IA - T1 Inteligência Artificial PUCRS

Frontend em terminal para o jogo da velha com classificador de estados por ML.

## Arquivos

```
tictactoe_ia/
├── jogo.py               # programa principal
├── dataset_treino.csv    # dataset de treino (mesmo dos notebooks)
├── dataset_testes.csv    # dataset de teste (mesmo dos notebooks)
├── requirements.txt
└── README.md
```

## Como rodar

1. **Instalar dependências**
   ```bash
   pip install -r requirements.txt
   ```

2. **Executar**
   ```bash
   python jogo.py
   ```

## Modelos disponíveis

| # | Modelo | Observação |
|---|--------|-----------|
| 1 | **KNN** (padrão) | Implementação manual, melhor K escolhido automaticamente via teste |
| 2 | **Árvore de Decisão** | sklearn, criterion=entropy |

O KNN é escolhido como padrão por ser implementado do zero (sem sklearn), alinhado com o espírito do trabalho.

## Classes que a IA classifica

| Classe interna | Exibição |
|---|---|
| `Tem jogo` | Tem Jogo |
| `Possibilidade_fim` | Possibilidade de Fim de Jogo |
| `Empate` | Empate |
| `Fim_O_Vence` | O Vence |
| `Fim_X_Vence` | X Vence |

## Regras do enunciado implementadas

- **IA erra o fim real → jogo encerrado imediatamente**
- **IA detecta fim incorretamente → jogo continua normalmente**

## Score

Após cada jogada (humano ou máquina) o estado é enviado à IA.
O programa exibe: previsão da IA, estado real, acerto/erro, e acurácia acumulada.
