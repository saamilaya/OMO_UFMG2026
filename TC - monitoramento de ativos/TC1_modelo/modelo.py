import math
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pathlib import Path

def _construir_modelo(d, S=8, eta=0.2, delta=1e-3):
    n, m = d.shape                       # d: matriz 125 x 14, em metros inteiros
    carga_min = math.ceil(eta * n / S)   # 4
    M = pyo.ConcreteModel()
    M.I, M.J, M.K = pyo.RangeSet(n), pyo.RangeSet(m), pyo.RangeSet(S)

    # variaveis
    M.x = pyo.Var(M.I, M.J, within=pyo.Binary)
    M.y = pyo.Var(M.J, M.K, within=pyo.Binary)
    M.h = pyo.Var(M.I, M.K, within=pyo.Binary)
    M.z = pyo.Var(M.K, within=pyo.Binary)

    # funções objetivo
    M.f1 = pyo.Expression(expr=sum(d[i-1, j-1] * M.x[i, j] for i in M.I for j in M.J))
    M.f2 = pyo.Expression(expr=sum(M.z[k] for k in M.K))

    # restrições
    M.c1 = pyo.Constraint(M.K, rule=lambda M, k: sum(M.y[j, k] for j in M.J) == M.z[k])
    M.c2 = pyo.Constraint(M.I, rule=lambda M, i: sum(M.x[i, j] for j in M.J) == 1)
    M.c3 = pyo.Constraint(M.I, M.J, rule=lambda M, i, j: M.x[i, j] <= sum(M.y[j, k] for k in M.K))
    M.c4 = pyo.Constraint(M.I, rule=lambda M, i: sum(M.h[i, k] for k in M.K) == 1)
    M.c5 = pyo.Constraint(M.I, M.J, M.K, rule=lambda M, i, j, k: M.h[i, k] + M.x[i, j] <= 1 + M.y[j, k])
    M.c6 = pyo.Constraint(M.K, rule=lambda M, k: sum(M.h[i, k] for i in M.I) >= carga_min * M.z[k])
    M.c7 = pyo.Constraint(M.I, M.K, rule=lambda M, i, k: M.h[i, k] <= M.z[k])
    M.sim = pyo.Constraint(pyo.RangeSet(S - 1), rule=lambda M, k: M.z[k] >= M.z[k + 1])  # simetria

    # parâmetros mutáveis e folga do AUGMECON
    M.eps = pyo.Param(initialize=S, mutable=True)
    M.r2 = pyo.Param(initialize=1, mutable=True)
    M.f1_lim = pyo.Param(initialize=1e12, mutable=True)
    M.delta = pyo.Param(initialize=delta)
    M.s = pyo.Var(within=pyo.NonNegativeReals)
    M.ceps = pyo.Constraint(expr=M.f2 + M.s == M.eps)
    M.cf1 = pyo.Constraint(expr=M.f1 <= M.f1_lim)       

    # três objetivos alternativos (só um ativo por vez)
    M.obj_f1 = pyo.Objective(expr=M.f1, sense=pyo.minimize)
    M.obj_f2 = pyo.Objective(expr=M.f2, sense=pyo.minimize)
    M.obj_aug = pyo.Objective(expr=M.f1 - M.delta * M.s / M.r2, sense=pyo.minimize)
    return M

def _resolver(M, objetivo, gap=0.0, tempo=600):
    for o in (M.obj_f1, M.obj_f2, M.obj_aug):
        o.deactivate()
    objetivo.activate()
    solver = pyo.SolverFactory('appsi_highs')
    solver.highs_options = {'mip_rel_gap': gap, 'time_limit': tempo}   # confirmar nomes
    return solver.solve(M)

def _payoff(M, S=8, tau=0.5):
    M.eps = S; M.cf1.deactivate()
    _resolver(M, M.obj_f1);  f1_min = pyo.value(M.f1)
    M.f1_lim = f1_min + tau; M.cf1.activate()
    _resolver(M, M.obj_f2);  f2_max = round(pyo.value(M.f2))   # lexicográfica
    M.cf1.deactivate()
    _resolver(M, M.obj_f2);  f2_min = round(pyo.value(M.f2))
    return f1_min, f2_min, f2_max

def _augmecon(M, f2_min, f2_max):
    M.r2 = max(f2_max - f2_min, 1)
    pontos = []
    for e in range(f2_min, f2_max + 1):
        M.eps = e
        _resolver(M, M.obj_aug)
        bases = [j for j in M.J if sum(pyo.value(M.y[j, k]) for k in M.K) > 0.5]
        atrib = {i: next(j for j in M.J if pyo.value(M.x[i, j]) > 0.5) for i in M.I}
        pontos.append(dict(eps=e, f1=pyo.value(M.f1), f2=round(pyo.value(M.f2)),
                           bases=bases, atrib=atrib))
    return pontos


if __name__ == "__main__":

    pasta = Path.cwd()
    arquivo = pasta / 'probdata.csv'

    df_dados = pd.read_csv(arquivo, sep = ';', decimal = ',', header= None,
                           names=['lat_base', 'lon_base', 'lat_ativo', 'lon_ativo', 'dist'])
    
    # identifico tanto as bases quanto os ativos de [0,13] e [0,24] respectivamente.
    df_dados['base_id'] = df_dados.groupby(['lat_base', 'lon_base'], sort=False).ngroup()
    df_dados['ativo_id'] = df_dados.groupby(['lat_ativo', 'lon_ativo'], sort=False).ngroup()

    df_dados.to_csv("identificadores.csv", sep=";", decimal=",", index=True)

    # construção da matriz ativos x base
    matriz_d = df_dados.pivot(index='ativo_id', columns='base_id', values='dist').to_numpy()
    matriz_d = np.rint(matriz_d * 1000)     # Para versão 2 com a distancia em metros
    
    M = _construir_modelo(matriz_d)
    f1_min, f2_min, f2_max = _payoff(M)
    print(f1_min, f2_min, f2_max)
    pontos = _augmecon(M, f2_min, f2_max)
    for p in pontos:
        print(p['eps'], p['f2'], p['f1'], p['bases'])

    df = pd.DataFrame.from_dict(pontos)
    df.to_csv("resultados_V2.csv", sep=";", decimal=",", index=True)