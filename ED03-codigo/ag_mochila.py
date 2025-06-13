import random
import numpy as np
from deap import base, creator, tools, algorithms
import time
import matplotlib.pyplot as plt
import os

# ==============================================
# 1. CONFIGURAÇÃO DO PROBLEMA DA MOCHILA
# ==============================================
def carregar_problema_mochila(caminho_arquivo):
    try:
        with open(caminho_arquivo, 'r') as arquivo:
            linhas = [linha.strip() for linha in arquivo if linha.strip()]
            n_itens = int(linhas[0])
            capacidade = int(linhas[1])
            valores = []
            pesos = []
            for linha in linhas[2:2+n_itens]:
                partes = linha.split()
                if len(partes) >= 2:
                    valores.append(int(partes[0]))
                    pesos.append(int(partes[1]))
        return n_itens, capacidade, valores, pesos
    except (FileNotFoundError, ValueError, IndexError) as e:
        n_itens = 10
        capacidade = 30
        valores = [random.randint(5, 20) for _ in range(n_itens)]
        pesos = [random.randint(1, 15) for _ in range(n_itens)]
        print(f"Erro ao carregar arquivo ({e}). Usando dados aleatórios.")
        return n_itens, capacidade, valores, pesos

n_itens, capacidade, valores, pesos = carregar_problema_mochila("knapsack_instance.txt")

# ==============================================
# 2. CONFIGURAÇÃO DO ALGORITMO GENÉTICO
# ==============================================
creator.create("AptidaoMax", base.Fitness, weights=(1.0,))
creator.create("Individuo", list, fitness=creator.AptidaoMax)

def avaliar_mochila(individuo):
    valor_total = sum(v for v, g in zip(valores, individuo) if g == 1)
    peso_total = sum(w for w, g in zip(pesos, individuo) if g == 1)
    if peso_total > capacidade:
        return 0,
    return valor_total,

def crossover_um_ponto(ind1, ind2):
    ponto = random.randint(1, len(ind1) - 1)
    ind1[ponto:], ind2[ponto:] = ind2[ponto:], ind1[ponto:]
    return ind1, ind2

def crossover_dois_pontos(ind1, ind2):
    size = min(len(ind1), len(ind2))
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
    return ind1, ind2

def crossover_uniforme(ind1, ind2):
    for i in range(len(ind1)):
        if random.random() < 0.5:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2

def mutacao_bit_flip(individuo, indpb):
    for i in range(len(individuo)):
        if random.random() < indpb:
            individuo[i] = 1 - individuo[i]
    return individuo,

def individuo_viavel(icls, n_itens, capacidade, pesos):
    while True:
        individuo = [random.randint(0, 1) for _ in range(n_itens)]
        peso_total = sum(w for w, g in zip(pesos, individuo) if g == 1)
        if peso_total <= capacidade:
            return icls(individuo)

def inicializacao_heuristica(icls):
    ratio = sorted([(i, valores[i] / pesos[i]) for i in range(n_itens)], key=lambda x: x[1], reverse=True)
    individuo = [0] * n_itens
    peso_total = 0
    for i, _ in ratio:
        if peso_total + pesos[i] <= capacidade:
            individuo[i] = 1
            peso_total += pesos[i]
    return icls(individuo)

# ==============================================
# 3. FUNÇÕES DE PLOTAGEM
# ==============================================
def plotar_evolucao_aptidao(todas_aptidoes):
    plt.figure(figsize=(10, 6))
    for config, valores_maximos in todas_aptidoes.items():
        plt.plot(range(1, len(valores_maximos)+1), valores_maximos, label=config, linewidth=2)
    plt.xlabel("Geração")
    plt.ylabel("Aptidão Máxima")
    plt.title("Evolução da Aptidão Máxima por Configuração")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("aptidao_comparacao.png")
    plt.show()

def plotar_tempo_execucao(tempos):
    configuracoes = list(tempos.keys())
    valores = list(tempos.values())
    plt.figure(figsize=(8, 5))
    bars = plt.bar(configuracoes, valores, color=plt.cm.tab10.colors)
    plt.title("Tempo de Execução por Configuração")
    plt.ylabel("Tempo (s)")
    plt.ylim(0, max(valores) + 2)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f"{yval:.1f}s", ha='center', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("tempo_execucao_comparacao.png")
    plt.show()

# ==============================================
# 4. EXECUÇÃO DO ALGORITMO
# ==============================================
def executar_configuracao(nome_config, crossover, mutpb, inicializador, convergencia=False):
    toolbox = base.Toolbox()
    toolbox.register("evaluate", avaliar_mochila)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutacao_bit_flip, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    if inicializador == inicializacao_heuristica:
        toolbox.register("individual", lambda: inicializador(creator.Individuo))
    else:
        toolbox.register("individual", inicializador)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    populacao = toolbox.population(n=50)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    inicio = time.time()
    NGEN = 100
    CXPB = 0.7
    MUTPB = mutpb
    valores_max = []
    n_convergencia = 20
    estagnado = 0
    melhor_aptidao = 0

    for gen in range(NGEN):
        offspring = algorithms.varAnd(populacao, toolbox, CXPB, MUTPB)
        fits = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit
        populacao = toolbox.select(offspring, k=len(populacao))
        record = stats.compile(populacao)
        valores_max.append(record['max'])

        if convergencia:
            if record['max'] <= melhor_aptidao:
                estagnado += 1
            else:
                melhor_aptidao = record['max']
                estagnado = 0
            if estagnado >= n_convergencia:
                break

    tempo = time.time() - inicio
    melhor = tools.selBest(populacao, 1)[0]
    print(f"{nome_config}: Valor={melhor.fitness.values[0]}, Peso={sum(w for w, g in zip(pesos, melhor) if g==1)}, Tempo={tempo:.2f}s")
    return valores_max, tempo

def main():
    configuracoes = {
        "Base": (crossover_dois_pontos, 0.2, lambda: tools.initRepeat(creator.Individuo, lambda: random.randint(0, 1), n=n_itens), False),
        "Uniforme": (crossover_uniforme, 0.2, lambda: tools.initRepeat(creator.Individuo, lambda: random.randint(0, 1), n=n_itens), False),
        "UmPonto": (crossover_um_ponto, 0.2, lambda: tools.initRepeat(creator.Individuo, lambda: random.randint(0, 1), n=n_itens), False),
        "AltaMutacao": (crossover_dois_pontos, 0.5, lambda: tools.initRepeat(creator.Individuo, lambda: random.randint(0, 1), n=n_itens), False),
        "HeuristicaConverg": (crossover_dois_pontos, 0.2, inicializacao_heuristica, True)
    }

    tempos_execucao = {}
    historico_aptidoes = {}

    for nome, (crossover, mutpb, init_func, use_convergencia) in configuracoes.items():
        aptidoes, tempo = executar_configuracao(nome, crossover, mutpb, init_func, convergencia=use_convergencia)
        tempos_execucao[nome] = tempo
        historico_aptidoes[nome] = aptidoes

    plotar_evolucao_aptidao(historico_aptidoes)
    plotar_tempo_execucao(tempos_execucao)

if __name__ == "__main__":
    main()
