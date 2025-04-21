import heapq
from collections import deque

class Puzzle8:
    def __init__(self, estado):
        self.estado = tuple(estado)  # Representação imutável
        self.goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)

    def is_goal(self):
        return self.estado == self.goal

    def get_neighbors(self):
        vizinhos = []
        idx = self.estado.index(0)
        linha, coluna = divmod(idx, 3)
        movimentos = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # cima, baixo, esquerda, direita

        for dl, dc in movimentos:
            nova_linha, nova_coluna = linha + dl, coluna + dc
            if 0 <= nova_linha < 3 and 0 <= nova_coluna < 3:
                novo_idx = nova_linha * 3 + nova_coluna
                novo_estado = list(self.estado)
                novo_estado[idx], novo_estado[novo_idx] = novo_estado[novo_idx], novo_estado[idx]
                vizinhos.append(Puzzle8(novo_estado))
        return vizinhos

    def __hash__(self):
        return hash(self.estado)

    def __eq__(self, other):
        return self.estado == other.estado

    def __lt__(self, other):
        return self.estado < other.estado

    def __str__(self):
        return str(self.estado)

# Heurística: número de peças fora do lugar
def heuristica_mal_colocado(puzzle):
    return sum(1 for i, v in enumerate(puzzle.estado) if v != 0 and v != puzzle.goal[i])

# Heurística: distância de Manhattan
def heuristica_manhattan(puzzle):
    distancia = 0
    for idx, val in enumerate(puzzle.estado):
        if val == 0:
            continue
        goal_idx = puzzle.goal.index(val)
        linha_atual, col_atual = divmod(idx, 3)
        linha_meta, col_meta = divmod(goal_idx, 3)
        distancia += abs(linha_atual - linha_meta) + abs(col_atual - col_meta)
    return distancia

# Busca em Largura (BFS)
def bfs(inicial):
    visitado = set()
    fila = deque([(inicial, [])])
    while fila:
        atual, caminho = fila.popleft()
        if atual in visitado:
            continue
        visitado.add(atual)
        if atual.is_goal():
            return caminho + [atual]
        for vizinho in atual.get_neighbors():
            fila.append((vizinho, caminho + [atual]))
    return None

# Busca em Profundidade (DFS)
def dfs(inicial, limite=30):
    visitado = set()
    pilha = [(inicial, [])]
    while pilha:
        atual, caminho = pilha.pop()
        if atual in visitado or len(caminho) > limite:
            continue
        visitado.add(atual)
        if atual.is_goal():
            return caminho + [atual]
        for vizinho in atual.get_neighbors():
            pilha.append((vizinho, caminho + [atual]))
    return None

# Busca Gulosa
def gulosa(inicial):
    visitado = set()
    heap = [(heuristica_manhattan(inicial), inicial, [])]
    while heap:
        _, atual, caminho = heapq.heappop(heap)
        if atual in visitado:
            continue
        visitado.add(atual)
        if atual.is_goal():
            return caminho + [atual]
        for vizinho in atual.get_neighbors():
            heapq.heappush(heap, (heuristica_manhattan(vizinho), vizinho, caminho + [atual]))
    return None

# A*
def a_star(inicial):
    visitado = set()
    heap = [(heuristica_manhattan(inicial), 0, inicial, [])]
    while heap:
        f, g, atual, caminho = heapq.heappop(heap)
        if atual in visitado:
            continue
        visitado.add(atual)
        if atual.is_goal():
            return caminho + [atual]
        for vizinho in atual.get_neighbors():
            novo_g = g + 1
            novo_f = novo_g + heuristica_manhattan(vizinho)
            heapq.heappush(heap, (novo_f, novo_g, vizinho, caminho + [atual]))
    return None
