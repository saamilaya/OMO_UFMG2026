import math
import time
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.contrib.appsi.solvers import Highs
from pyomo.contrib.appsi.base import TerminationCondition as TC
from pathlib import Path



def _construir_modelo(d, Smax=8, eta=0.2, delta=1e-3):
    n, m = d.shape                        # d: matriz 125 x 14; n: ativos e m: bases
    M = pyo.ConcreteModel()
    M.I, M.J, M.K = pyo.RangeSet(n), pyo.RangeSet(m), pyo.RangeSet(Smax)

    # variaveis
    M.x = pyo.Var(M.I, M.J, within=pyo.Binary)  # ativo i alocado à base j
    M.y = pyo.Var(M.J, M.K, within=pyo.Binary)  # base j ocupada pela equipe k
    M.h = pyo.Var(M.I, M.K, within=pyo.Binary)  # ativo i atribuido à equipe k
    M.z = pyo.Var(M.K, within=pyo.Binary)       # equipe k foi contratada

    # funcoes objetivo
    M.f1 = pyo.Expression(expr=sum(d[i-1, j-1] * M.x[i, j] for i in M.I for j in M.J))  # Somatório da distancia entre a base e os ativos alocados
    M.f2 = pyo.Expression(expr=sum(M.z[k] for k in M.K))                                # Somatório das equipes contratadas

    # Restrições

    # c1: Cada equipe contratada deve ser alocada a exatamente uma base
    def c1_rule(M, k):
        return sum(M.y[j, k] for j in M.J) == M.z[k]

    M.c1 = pyo.Constraint(M.K, rule=c1_rule)


    # c2: Cada ativo é atribuido a exatamente uma base
    def c2_rule(M, i):
        return sum(M.x[i, j] for j in M.J) == 1

    M.c2 = pyo.Constraint(M.I, rule=c2_rule)


    # c3: Um ativo só pode ser atribuido a uma base se ela estiver ocupada
    def c3_rule(M, i, j):
        return M.x[i, j] <= sum(M.y[j, k] for k in M.K)

    M.c3 = pyo.Constraint(M.I, M.J, rule=c3_rule)


    # c4: Cada ativo deve ser atribuido a exatamente uma equipe
    def c4_rule(M, i):
        return sum(M.h[i, k] for k in M.K) == 1

    M.c4 = pyo.Constraint(M.I, rule=c4_rule)


    # c5: Compatibilidade entre a alocação do ativo, da equipe e da base
    # Um ativo i alocado à base j só pode ser atribuido a uma equipe k se ela estiver ocupando a mesma base j
    def c5_rule(M, i, j, k):
        return M.h[i, k] + M.x[i, j] <= 1 + M.y[j, k]

    M.c5 = pyo.Constraint(M.I, M.J, M.K, rule=c5_rule)


    # c6: Carga mínima de ativos por equipe. eta: 0.2
    # Cada equipe empregada deve manter pelo menos eta*n/s ativos, em que s = f2 é o
    # número de equipes EMPREGADAS (e não o máximo Smax). Como s é variável, a carga

    carga_min = {s: math.ceil(eta * n / s) for s in range(1, Smax + 1)}

    # Linearização: com a quebra de simetria (c8), as equipes empregadas são as
    # primeiras, e (z[s] - z[s+1]) vale 1 só para s = f2 (com z[Smax+1] = 0).
    # Assim, para a equipe k:
    #   - se z[k] = 1, a soma abaixo vale exatamente L(f2);
    #   - se z[k] = 0, todos os z[s] com s >= k são nulos e a soma vale 0.
    def z_ext(M, s):
        return M.z[s] if s <= Smax else 0

    def c6_rule(M, k):
        exigido = sum(carga_min[s] * (z_ext(M, s) - z_ext(M, s + 1)) for s in range(k, Smax + 1))
        return sum(M.h[i, k] for i in M.I) >= exigido

    M.c6 = pyo.Constraint(M.K, rule=c6_rule)


    # c7: Um ativo só pode ser alocado à equipe k se a mesma estiver contratada
    def c7_rule(M, i, k):
        return M.h[i, k] <= M.z[k]

    M.c7 = pyo.Constraint(M.I, M.K, rule=c7_rule)


    # c8: simetria. Ou seja, evita repetições. Também é usada na linearização de c6. Ex. para sum(z) = 3; z = (1, 1, 1, 0, 0, 0, 0, 0)
    # sem precisar testar outras configurações como z = (0,0,1,0,1,0,1,0)
    def sim_rule(M, k):
        return M.z[k] >= M.z[k + 1]

    M.sim = pyo.Constraint(pyo.RangeSet(Smax - 1), rule=sim_rule)

    # AUGMECON: f2 + folga = eps, folga >= 0 (folga, pois f2 e minimizado)
    M.eps = pyo.Param(initialize=Smax, mutable=True)
    M.r2 = pyo.Param(initialize=1, mutable=True)
    M.f1_lim = pyo.Param(initialize=1e12, mutable=True)
    M.delta = pyo.Param(initialize=delta)
    M.folga = pyo.Var(within=pyo.NonNegativeReals)              # folga da restricao epsilon
    M.ceps = pyo.Constraint(expr=M.f2 + M.folga == M.eps)
    M.cf1 = pyo.Constraint(expr=M.f1 <= M.f1_lim)

    # objetivos alternativos (apenas um objetivo ativo por vez)
    M.obj_f1 = pyo.Objective(expr=M.f1, sense=pyo.minimize)
    M.obj_f2 = pyo.Objective(expr=M.f2, sense=pyo.minimize)
    # minimizar f1 e, em segundo nivel, maximizar a folga normalizada folga/r2
    M.obj_aug = pyo.Objective(expr=M.f1 - M.delta * M.folga / M.r2, sense=pyo.minimize)
    return M


def _resolver(M, objetivo, tempo=600):
    """
    Resolve o modelo usando 'objetivo' como funcao objetivo e
    retorna um dicionario 'solucao' com os valores encontrados.
    """
    # O Pyomo aceita apenas um objetivo ativo: desliga os tres e liga o escolhido
    M.obj_f1.deactivate()
    M.obj_f2.deactivate()
    M.obj_aug.deactivate()
    objetivo.activate()

    solver = Highs()
    solver.highs_options = {
        'mip_rel_gap': 0.0,    # exige o otimo exato (sem tolerancia de gap)
        'mip_abs_gap': 0.0,
        'time_limit': tempo,   # segundos
    }

    resultado = solver.solve(M)
    status = resultado.termination_condition

    if status != TC.optimal:
        # Ex.: estourou o limite de tempo; a solucao carregada nao e otima
        raise RuntimeError(f'solver terminou com status {status}')

    # Matrizes de decisao. O solver devolve binarias como 0.9999999 ou 1e-10,
    # por isso arredondamos para inteiro (0 ou 1).
    X = np.array([[pyo.value(M.x[i, j]) for j in M.J] for i in M.I]).round().astype(int)  # n x m
    Y = np.array([[pyo.value(M.y[j, k]) for k in M.K] for j in M.J]).round().astype(int)  # m x Smax
    H = np.array([[pyo.value(M.h[i, k]) for k in M.K] for i in M.I]).round().astype(int)  # n x Smax

    solucao = {
        'f1': pyo.value(M.f1),
        'f2': round(pyo.value(M.f2)),
        'folga': pyo.value(M.folga),
        'X': X,
        'Y': Y,
        'H': H,
    }
    return solucao


def _payoff(M, Smax=8, tol_f1=1e-6):
    """
    Tabela payoff lexicografica (Mavrotas 2009, sec. 3.1).
    Fornece os extremos da fronteira: (f1_min, f2_max) e (f1_max, f2_min).
    """
    # Com eps = Smax a restricao f2 + folga = eps nao limita nada, pois f2 <= Smax sempre.
    # A restricao auxiliar cf1 (f1 fixo no otimo) comeca desligada.
    M.eps = Smax
    M.cf1.deactivate()

    # Linha 1, etapa 1: menor distancia possivel
    solucao = _resolver(M, M.obj_f1)
    f1_min = solucao['f1']

    # Linha 1, etapa 2: entre as solucoes com f1 = f1_min, a que usa menos equipes.
    # Fixamos f1 no otimo com a restricao f1 <= f1_min + tol e minimizamos f2.
    M.f1_lim = f1_min + tol_f1
    M.cf1.activate()
    solucao = _resolver(M, M.obj_f2)
    f2_max = solucao['f2']
    M.cf1.deactivate()

    # Linha 2, etapa 1: menor numero de equipes possivel
    solucao = _resolver(M, M.obj_f2)
    f2_min = solucao['f2']

    # Linha 2, etapa 2: entre as solucoes com f2 = f2_min, a de menor distancia.
    # A propria restricao epsilon fixa f2: com eps = f2_min, f2 + folga = f2_min e folga >= 0
    # obrigam f2 = f2_min.
    M.eps = f2_min
    solucao = _resolver(M, M.obj_f1)
    f1_max = solucao['f1']

    return {'f1_min': f1_min, 'f1_max': f1_max, 'f2_min': f2_min, 'f2_max': f2_max}


def _augmecon(M, payoff, passo=1):
    """
    AUGMECON de Mavrotas (2009), para minimizacao:
    eps percorre TODOS os pontos da grade de f2, do pior (f2_max) ao melhor (f2_min)
    valor, sem salto (bypass). Se a solucao de uma iteracao tiver folga > 0, as
    iteracoes seguintes repetem a mesma solucao; ela e registrada de novo e as
    repeticoes sao identificadas no print e removidas no final.
    Como eps >= f2_min, todo subproblema e viavel e a saida antecipada nao ocorre.
    """
    f2_min, f2_max = payoff['f2_min'], payoff['f2_max']
    M.r2 = max(f2_max - f2_min, 1)
    pontos = []
    for e in range(f2_max, f2_min - 1, -passo):
        M.eps = e
        solucao = _resolver(M, M.obj_aug)

        X, Y, H = solucao['X'], solucao['Y'], solucao['H']

        # Organizacao da solucao (indices somados de 1 para coincidir com o modelo)
        bases = [j + 1 for j in range(Y.shape[0]) if Y[j, :].sum() > 0]   # bases ocupadas
        atrib = {i + 1: int(X[i, :].argmax()) + 1 for i in range(X.shape[0])}   # base de cada ativo
        equipes = {}
        for k in range(Y.shape[1]):
            if Y[:, k].sum() == 1:                                         # equipe k contratada
                equipes[k + 1] = {
                    'base': int(Y[:, k].argmax()) + 1,                     # base que ocupa
                    'ativos': [i + 1 for i in range(H.shape[0]) if H[i, k] == 1],
                }

        # A solucao e repetida se tiver os mesmos (f1, f2) do ponto anterior
        repetida = len(pontos) > 0 and pontos[-1]['f2'] == solucao['f2'] \
                   and abs(pontos[-1]['f1'] - solucao['f1']) < 1e-6
        print(f"eps = {e}: f2 = {solucao['f2']}, f1 = {solucao['f1']:.2f}, "
              f"folga = {solucao['folga']:.3f}" + (" -> solucao repetida" if repetida else ""))

        pontos.append({
            'eps': e,
            'f1': solucao['f1'],
            'f2': solucao['f2'],
            'folga': solucao['folga'],
            'repetida': repetida,
            'bases': bases,
            'atrib': atrib,
            'equipes': equipes,
        })

    n_resolucoes = len(pontos)
    pontos = [p for p in pontos if not p['repetida']]    # fronteira sem repeticoes
    print(f"{n_resolucoes} resolucoes, {len(pontos)} pontos distintos")
    return pontos


if __name__ == "__main__":

    pasta = Path.cwd() / 'dados'
    arquivo = pasta / 'probdata.csv'

    df_dados = pd.read_csv(arquivo, sep=';', decimal=',', header=None,
                           names=['lat_base', 'lon_base', 'lat_ativo', 'lon_ativo', 'dist'])

    df_dados['base_id'] = df_dados.groupby(['lat_base', 'lon_base'], sort=False).ngroup()
    df_dados['ativo_id'] = df_dados.groupby(['lat_ativo', 'lon_ativo'], sort=False).ngroup()
    df_dados.to_csv("identificadores.csv", sep=";", decimal=",", index=True)

    # matriz ativos x bases em km (f1 real; f2 e o objetivo restrito e ja e inteiro)
    matriz_d = df_dados.pivot(index='ativo_id', columns='base_id', values='dist').to_numpy()

    M = _construir_modelo(matriz_d)
    inicio = time.time()
    payoff = _payoff(M)
    print('payoff:', payoff)
    pontos = _augmecon(M, payoff)
    fim = time.time()
    tempo_decorrido = fim - inicio
    print(f"Tempo de execução: {tempo_decorrido:.4f} segundos")

    for ponto in pontos:
        print(ponto['eps'], ponto['f2'], ponto['f1'], ponto['folga'], ponto['bases'])

    df = pd.DataFrame.from_dict(pontos)
    df.to_csv("resultados_augmecon2009.csv", sep=";", decimal=",", index=True)
