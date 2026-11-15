"""
Monitoramento de ativos: alocacao de equipes a bases e atribuicao de ativos.
Fronteira de Pareto exata via AUGMECON2 (Mavrotas & Florios, 2013).

Objetivos (ambos de minimizacao):
    f1 = distancia total ativo -> base da equipe responsavel (metros, inteiro)
    f2 = numero de equipes empregadas

Como f2 e inteiro e o problema e biobjetivo, o passo unitario sobre f2 e o
payoff lexicografico garantem a fronteira exata (Mavrotas & Florios, 2013, sec. 3.2).
"""
import math
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.contrib.appsi.solvers import Highs
from pyomo.contrib.appsi.base import TerminationCondition as TC
from pathlib import Path


def _construir_modelo(d, S=8, eta=0.2, delta=1e-3):
    n, m = d.shape                        # d: n x m, metros inteiros
    carga_min = math.ceil(eta * n / S)    # restricao 6 do enunciado, s = 8
    M = pyo.ConcreteModel()
    M.I, M.J, M.K = pyo.RangeSet(n), pyo.RangeSet(m), pyo.RangeSet(S)

    # variaveis (x, y, h conforme o enunciado; z marca equipe contratada)
    M.x = pyo.Var(M.I, M.J, within=pyo.Binary)
    M.y = pyo.Var(M.J, M.K, within=pyo.Binary)
    M.h = pyo.Var(M.I, M.K, within=pyo.Binary)
    M.z = pyo.Var(M.K, within=pyo.Binary)

    # funcoes objetivo
    M.f1 = pyo.Expression(expr=sum(int(d[i-1, j-1]) * M.x[i, j] for i in M.I for j in M.J))
    M.f2 = pyo.Expression(expr=sum(M.z[k] for k in M.K))

    # restricoes 1 a 6 do enunciado (1 relaxada para equipes nao contratadas)
    M.c1 = pyo.Constraint(M.K, rule=lambda M, k: sum(M.y[j, k] for j in M.J) == M.z[k])
    M.c2 = pyo.Constraint(M.I, rule=lambda M, i: sum(M.x[i, j] for j in M.J) == 1)
    M.c3 = pyo.Constraint(M.I, M.J, rule=lambda M, i, j: M.x[i, j] <= sum(M.y[j, k] for k in M.K))
    M.c4 = pyo.Constraint(M.I, rule=lambda M, i: sum(M.h[i, k] for k in M.K) == 1)
    M.c5 = pyo.Constraint(M.I, M.J, M.K, rule=lambda M, i, j, k: M.h[i, k] + M.x[i, j] <= 1 + M.y[j, k])
    M.c6 = pyo.Constraint(M.K, rule=lambda M, k: sum(M.h[i, k] for i in M.I) >= carga_min * M.z[k])
    M.c7 = pyo.Constraint(M.I, M.K, rule=lambda M, i, k: M.h[i, k] <= M.z[k])        # redundante, ajuda o B&B
    M.sim = pyo.Constraint(pyo.RangeSet(S - 1), rule=lambda M, k: M.z[k] >= M.z[k + 1])  # simetria

    # AUGMECON2: f2 + s = eps, s >= 0 (folga, pois f2 e minimizado)
    M.eps = pyo.Param(initialize=S, mutable=True)
    M.r2 = pyo.Param(initialize=1, mutable=True)
    M.f1_lim = pyo.Param(initialize=1e12, mutable=True)
    M.f2_lim = pyo.Param(initialize=S, mutable=True)
    M.delta = pyo.Param(initialize=delta)
    M.s = pyo.Var(within=pyo.NonNegativeReals)
    M.ceps = pyo.Constraint(expr=M.f2 + M.s == M.eps)
    M.cf1 = pyo.Constraint(expr=M.f1 <= M.f1_lim)
    M.cf2 = pyo.Constraint(expr=M.f2 <= M.f2_lim)

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


def _payoff(M, S=8):
    """Tabela payoff lexicografica (Mavrotas 2009, sec. 3.1). Para p = 2 ela da o nadir exato."""
    M.eps = S
    M.cf1.deactivate(); M.cf2.deactivate()

    # linha 1: min f1, depois min f2 com f1 fixo no otimo (f1 inteiro -> folga 0.5 basta)
    _resolver(M, M.obj_f1);  f1_min = round(pyo.value(M.f1))
    M.f1_lim = f1_min + 0.5; M.cf1.activate()
    _resolver(M, M.obj_f2);  f2_max = round(pyo.value(M.f2))
    M.cf1.deactivate()

    # linha 2: min f2, depois min f1 com f2 fixo no otimo
    _resolver(M, M.obj_f2);  f2_min = round(pyo.value(M.f2))
    M.f2_lim = f2_min; M.cf2.activate()
    _resolver(M, M.obj_f1);  f1_max = round(pyo.value(M.f1))
    M.cf2.deactivate()

    return dict(f1_min=f1_min, f1_max=f1_max, f2_min=f2_min, f2_max=f2_max)


def _extrair(M, e):
    bases = [j for j in M.J if sum(pyo.value(M.y[j, k]) for k in M.K) > 0.5]
    atrib = {i: next(j for j in M.J if pyo.value(M.x[i, j]) > 0.5) for i in M.I}
    equipes = {k: dict(base=next(j for j in M.J if pyo.value(M.y[j, k]) > 0.5),
                       ativos=[i for i in M.I if pyo.value(M.h[i, k]) > 0.5])
               for k in M.K if pyo.value(M.z[k]) > 0.5}
    return dict(eps=e, f1=round(pyo.value(M.f1)), f2=round(pyo.value(M.f2)),
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

    pasta = Path.cwd()
    arquivo = pasta / 'probdata.csv'

    df_dados = pd.read_csv(arquivo, sep=';', decimal=',', header=None,
                           names=['lat_base', 'lon_base', 'lat_ativo', 'lon_ativo', 'dist'])

    df_dados['base_id'] = df_dados.groupby(['lat_base', 'lon_base'], sort=False).ngroup()
    df_dados['ativo_id'] = df_dados.groupby(['lat_ativo', 'lon_ativo'], sort=False).ngroup()
    df_dados.to_csv("identificadores.csv", sep=";", decimal=",", index=True)

    # matriz ativos x bases em metros inteiros (coeficientes inteiros de f1)
    matriz_d = df_dados.pivot(index='ativo_id', columns='base_id', values='dist').to_numpy()
    matriz_d = np.rint(matriz_d * 1000).astype(int)

    M = _construir_modelo(matriz_d)
    payoff = _payoff(M)
    print('payoff:', payoff)
    pontos = _augmecon2(M, payoff)
    for p in pontos:
        print(p['eps'], p['f2'], p['f1'], p['folga'], p['bases'])

    df = pd.DataFrame.from_dict(pontos)
    df.to_csv("resultados_V2.csv", sep=";", decimal=",", index=True)
