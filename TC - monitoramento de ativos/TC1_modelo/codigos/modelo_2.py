import math
import time
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.contrib.appsi.solvers import Highs
from pyomo.contrib.appsi.base import TerminationCondition as TC
from pathlib import Path



def _construir_modelo(d, S=8, eta=0.2, delta=1e-3):
    n, m = d.shape                        # d: matriz 125 x 14; n: ativos e m: bases                       
    M = pyo.ConcreteModel()
    M.I, M.J, M.K = pyo.RangeSet(n), pyo.RangeSet(m), pyo.RangeSet(S)

    # variaveis 
    M.x = pyo.Var(M.I, M.J, within=pyo.Binary)  # ativo i alocado à base j 
    M.y = pyo.Var(M.J, M.K, within=pyo.Binary)  # base k ocupada pela equipe k
    M.h = pyo.Var(M.I, M.K, within=pyo.Binary)  # ativo i atribuido à equipe k
    M.z = pyo.Var(M.K, within=pyo.Binary)       # equipe k foi contratada

    # funcoes objetivo
    M.f1 = pyo.Expression(expr=sum(d[i-1, j-1] * M.x[i, j] for i in M.I for j in M.J))  # Somatório da distancia entre a bese e os ativos alocados
    M.f2 = pyo.Expression(expr=sum(M.z[k] for k in M.K))                                # Somatório das equipes contratadas

    # Restrições 

    # c1: Cada equipe contratada deve ser alocação a exatamente uma base
    def c1_rule(M, k):
        return sum(M.y[j, k] for j in M.J) == M.z[k]

    M.c1 = pyo.Constraint(M.K, rule=c1_rule)


    # c2: Cada ativo é atribuido a exatamente uma base
    def c2_rule(M, i):
        return sum(M.x[i, j] for j in M.J) == 1

    M.c2 = pyo.Constraint(M.I, rule=c2_rule)


    # c3: Um ativo só pode ser atibuido a uma base se ela estiver ocupada 
    def c3_rule(M, i, j):
        return M.x[i, j] <= sum(M.y[j, k] for k in M.K)

    M.c3 = pyo.Constraint(M.I, M.J, rule=c3_rule)


    # c4: Cada ativo deve ser atribuido a exatamente uma equipe
    def c4_rule(M, i):
        return sum(M.h[i, k] for k in M.K) == 1

    M.c4 = pyo.Constraint(M.I, rule=c4_rule)


    # c5: Compatibilidade entre a alocação do ativo, da equipe e da base
    # Um ativo i alocado à base j só pode ser atribuido a uma equipe k se estiver ocupando a mesma base j 
    def c5_rule(M, i, j, k):
        return M.h[i, k] + M.x[i, j] <= 1 + M.y[j, k]

    M.c5 = pyo.Constraint(M.I, M.J, M.K, rule=c5_rule)


    # c6: Carga mínima de ativos por equipe. eta: 0.2
    carga_min = math.ceil(eta * n / S)

    def c6_rule(M, k):
        return sum(M.h[i, k] for i in M.I) >= carga_min * M.z[k]

    M.c6 = pyo.Constraint(M.K, rule=c6_rule)


    # c7: Um ativo só pode ser alocado à equipe k se a mesma estiver contratada
    def c7_rule(M, i, k):
        return M.h[i, k] <= M.z[k]

    M.c7 = pyo.Constraint(M.I, M.K, rule=c7_rule)


    # c8: simentria. Ou seja, evita repetições. Ex. para sum(z) = 3; z = (1, 1, 1, 0, 0, 0, 0, 0) 
    # sem precisar testar outras configurações como z = (0,0,1,0,1,0,1,0)
    def sim_rule(M, k):
        return M.z[k] >= M.z[k + 1]

    M.sim = pyo.Constraint(pyo.RangeSet(S - 1), rule=sim_rule)

    # AUGMECON2: f2 + s = eps, s >= 0 (folga, pois f2 e minimizado)
    M.eps = pyo.Param(initialize=S, mutable=True)
    M.r2 = pyo.Param(initialize=1, mutable=True)
    M.f1_lim = pyo.Param(initialize=1e12, mutable=True)
    M.delta = pyo.Param(initialize=delta)
    M.s = pyo.Var(within=pyo.NonNegativeReals)
    M.ceps = pyo.Constraint(expr=M.f2 + M.s == M.eps)
    M.cf1 = pyo.Constraint(expr=M.f1 <= M.f1_lim)

    # objetivos alternativos (um ativo por vez)
    M.obj_f1 = pyo.Objective(expr=M.f1, sense=pyo.minimize)
    M.obj_f2 = pyo.Objective(expr=M.f2, sense=pyo.minimize)
    # minimizar f1 e, em segundo nivel, maximizar a folga normalizada s/r2
    M.obj_aug = pyo.Objective(expr=M.f1 - M.delta * M.s / M.r2, sense=pyo.minimize)
    return M


def _resolver(M, objetivo, gap=0.0, tempo=600):
    for o in (M.obj_f1, M.obj_f2, M.obj_aug):
        o.deactivate()
    objetivo.activate()
    solver = Highs()
    solver.highs_options = {'mip_rel_gap': gap, 'mip_abs_gap': 0.0, 'time_limit': tempo}
    solver.config.load_solution = False
    res = solver.solve(M)
    if res.termination_condition == TC.optimal:
        solver.load_vars()
        return True
    if res.termination_condition in (TC.infeasible, TC.infeasibleOrUnbounded):
        return False
    raise RuntimeError(f'solver terminou com {res.termination_condition} (limite de tempo?)')



def _payoff(M, S=8, tol_f1=1e-6):
    """
    Tabela payoff lexicografica (Mavrotas 2009, sec. 3.1).
    Fornece os extremos da fronteira: (f1_min, f2_max) e (f1_max, f2_min).
    """
    # Com eps = S a restricao f2 + s = eps nao limita nada, pois f2 <= S sempre.
    # A restricao auxiliar cf1 (f1 fixo no otimo) comeca desligada.
    M.eps = S
    M.cf1.deactivate()

    # Linha 1, etapa 1: menor distancia possivel
    _resolver(M, M.obj_f1)
    f1_min = pyo.value(M.f1)

    # Linha 1, etapa 2: entre as solucoes com f1 = f1_min, a que usa menos equipes.
    # Fixamos f1 no otimo com a restricao f1 <= f1_min + tol e minimizamos f2.
    M.f1_lim = f1_min + tol_f1
    M.cf1.activate()
    _resolver(M, M.obj_f2)
    f2_max = round(pyo.value(M.f2))
    M.cf1.deactivate()

    # Linha 2, etapa 1: menor numero de equipes possivel
    _resolver(M, M.obj_f2)
    f2_min = round(pyo.value(M.f2))

    # Linha 2, etapa 2: entre as solucoes com f2 = f2_min, a de menor distancia.
    # A propria restricao epsilon fixa f2: com eps = f2_min, f2 + s = f2_min e s >= 0
    # obrigam f2 = f2_min.
    M.eps = f2_min
    _resolver(M, M.obj_f1)
    f1_max = pyo.value(M.f1)

    return {'f1_min': f1_min, 'f1_max': f1_max, 'f2_min': f2_min, 'f2_max': f2_max}


def _extrair(M, e):
    bases = [j for j in M.J if sum(pyo.value(M.y[j, k]) for k in M.K) > 0.5]
    atrib = {i: next(j for j in M.J if pyo.value(M.x[i, j]) > 0.5) for i in M.I}
    equipes = {k: dict(base=next(j for j in M.J if pyo.value(M.y[j, k]) > 0.5),
                       ativos=[i for i in M.I if pyo.value(M.h[i, k]) > 0.5])
               for k in M.K if pyo.value(M.z[k]) > 0.5}
    return dict(eps=e, f1=pyo.value(M.f1), f2=round(pyo.value(M.f2)),
                folga=pyo.value(M.s), bases=bases, atrib=atrib, equipes=equipes)


def _augmecon2(M, payoff, passo=1):
    """
    Laco de Mavrotas & Florios (2013), fig. 1, para minimizacao:
    eps percorre f2 do pior (f2_max) ao melhor (f2_min) valor, apertando a restricao.
    Apos cada solucao, b = int(s/passo) iteracoes seguintes repetiriam a mesma solucao
    (so a folga mudaria) e sao puladas. Saida antecipada se o problema ficar inviavel.
    """
    f2_min, f2_max = payoff['f2_min'], payoff['f2_max']
    M.r2 = max(f2_max - f2_min, 1)
    pontos = []
    e = f2_max
    while e >= f2_min:
        M.eps = e
        if not _resolver(M, M.obj_aug):
            break                                   # saida antecipada: inviavel
        p = _extrair(M, e)
        pontos.append(p)
        b = int(p['folga'] / passo + 1e-9)          # coeficiente de bypass
        e -= passo * (b + 1)
    return pontos


if __name__ == "__main__":

    pasta = Path.cwd() / 'dados'
    arquivo = pasta / 'probdata.csv'

    df_dados = pd.read_csv(arquivo, sep=';', decimal=',', header=None,
                           names=['lat_base', 'lon_base', 'lat_ativo', 'lon_ativo', 'dist'])

    df_dados['base_id'] = df_dados.groupby(['lat_base', 'lon_base'], sort=False).ngroup()
    df_dados['ativo_id'] = df_dados.groupby(['lat_ativo', 'lon_ativo'], sort=False).ngroup()
    df_dados.to_csv("identificadores.csv", sep=";", decimal=",", index=True)

    # matriz ativos x bases em km (f1 real; so f2 precisa ser inteiro no AUGMECON2)
    matriz_d = df_dados.pivot(index='ativo_id', columns='base_id', values='dist').to_numpy()

    M = _construir_modelo(matriz_d)
    inicio = time.time()
    payoff = _payoff(M)
    print('payoff:', payoff)
    pontos = _augmecon2(M, payoff)
    fim = time.time()
    tempo_decorrido = fim - inicio
    print(f"Tempo de execução: {tempo_decorrido:.4f} segundos")

    for p in pontos:
        print(p['eps'], p['f2'], p['f1'], p['folga'], p['bases'])

    df = pd.DataFrame.from_dict(pontos)
    df.to_csv("resultados_modelo2.csv", sep=";", decimal=",", index=True)
