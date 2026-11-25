"""
Entrega 3.v -- Tomada de decisao multiatributo.
Criterios (todos de MINIMIZACAO): f1 (distancia), f2 (equipes),
custo (instalacao das bases), robustez (impacto da falha da base mais carregada).
Metodos: AHP (pesos) + TOPSIS + PROMETHEE II, com comparacao.
"""
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Custo ficticio de instalacao/operacao das 14 bases (R$ mil), gerado uma unica
# vez com seed fixa (np.random.default_rng(7), uniforme 80-250) -- dado novo
# assumido pela equipe, como o enunciado permite. Indexado 1..14.
CUSTO_BASE = {
    1: 186.3, 2: 232.5, 3: 211.9, 4: 118.3, 5: 131.0, 6: 228.5, 7: 80.9,
    8: 219.6, 9: 215.5, 10: 159.5, 11: 131.5, 12: 127.3, 13: 123.3, 14: 155.7,
}

CRITERIOS = ["f1", "f2", "custo", "robustez"]


def _carregar_dados(pasta=None):
    pasta = Path(pasta or Path.cwd())
    pontos = pd.read_csv(pasta / "resultados_modelo2.csv", sep=";", decimal=",")
    ident = pd.read_csv(pasta / "identificadores.csv", sep=";", decimal=",")
    return pontos.sort_values("f2").reset_index(drop=True), ident, pasta


def _matriz_distancias(ident):
    piv = ident.pivot(index="ativo_id", columns="base_id", values="dist")
    piv = piv.sort_index().sort_index(axis=1)
    d = {}
    for ai, ativo_id in enumerate(piv.index, start=1):
        for bi, base_id in enumerate(piv.columns, start=1):
            d[(ai, bi)] = piv.loc[ativo_id, base_id]
    return d


def _robustez(linha, d):
    bases = ast.literal_eval(str(linha["bases"]))
    atrib = ast.literal_eval(str(linha["atrib"]))
    atrib = {int(k): int(v) for k, v in atrib.items()}

    if len(bases) <= 1:
        return None   # unica base: falha dela deixa tudo sem cobertura

    carga = {b: sum(1 for j in atrib.values() if j == b) for b in bases}
    base_critica = max(carga, key=carga.get)
    alternativas = [b for b in bases if b != base_critica]

    delta = 0.0
    for ativo, base in atrib.items():
        if base == base_critica:
            dist_original = d[(ativo, base_critica)]
            dist_nova = min(d[(ativo, b)] for b in alternativas)
            delta += dist_nova - dist_original
    return delta


def montar_matriz_decisao(pasta=None):
    pontos, ident, pasta = _carregar_dados(pasta)
    d = _matriz_distancias(ident)

    linhas = []
    for _, linha in pontos.iterrows():
        bases = ast.literal_eval(str(linha["bases"]))
        custo = sum(CUSTO_BASE[b] for b in bases)
        robustez = _robustez(linha, d)
        linhas.append({"eps": int(linha["eps"]), "f1": linha["f1"],
                        "f2": linha["f2"], "custo": custo, "robustez": robustez})

    df = pd.DataFrame(linhas)
    pior = df["robustez"].max()
    df["robustez"] = df["robustez"].fillna(pior * 1.5 if pd.notna(pior) else 1.0)
    return df, pasta

def pesos_ahp(ordem):
    """ordem: lista com os 4 critérios, do mais importante ao menos importante.
    Monta a matriz de julgamentos em cascata (quem está 1 posição mais abaixo
    na ordem recebe nota 2 na escala de Saaty, 2 posições recebe nota 4, 3
    posições recebe nota 6) e calcula os pesos pelo método do autovetor
    principal."""
    n = len(ordem)
    nota_por_distancia = {1: 2, 2: 4, 3: 6}
    A = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist = j - i
            A[i, j] = nota_por_distancia[dist] if dist > 0 else 1 / nota_por_distancia[-dist]

    autovalores, autovetores = np.linalg.eig(A)
    idx = np.argmax(autovalores.real)
    lam = autovalores[idx].real
    w = autovetores[:, idx].real
    w = w / w.sum()
    CI = (lam - n) / (n - 1)
    CR = CI / 0.90
    pesos = dict(zip(ordem, w))
    return {c: pesos[c] for c in CRITERIOS}, lam, CR


# cada cenário: o critério citado vem em 1º lugar; a ordem dos outros três
# segue a prioridade original (robustez > f1 > f2 > custo)
CENARIOS = {
    "custo dominante":     ["custo", "robustez", "f1", "f2"],
    "distância (f1) dominante": ["f1", "robustez", "f2", "custo"],
    "equipes (f2) dominante":   ["f2", "robustez", "f1", "custo"],
    "robustez dominante (atual)": ["robustez", "f1", "f2", "custo"],
}



def topsis(df, pesos):
    X = df[CRITERIOS].to_numpy(dtype=float)
    w = np.array([pesos[c] for c in CRITERIOS])
    R = X / np.sqrt((X ** 2).sum(axis=0))
    V = R * w
    ideal_pos, ideal_neg = V.min(axis=0), V.max(axis=0)
    s_pos = np.sqrt(((V - ideal_pos) ** 2).sum(axis=1))
    s_neg = np.sqrt(((V - ideal_neg) ** 2).sum(axis=1))
    out = df.copy()
    out["topsis_c"] = s_neg / (s_pos + s_neg)
    return out.sort_values("topsis_c", ascending=False).reset_index(drop=True)


def promethee2(df, pesos):
    X = df[CRITERIOS].to_numpy(dtype=float)
    w = np.array([pesos[c] for c in CRITERIOS])
    n = len(X)
    ranges = X.max(axis=0) - X.min(axis=0)
    ranges[ranges == 0] = 1.0

    pi = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            dif = (X[b] - X[a]) / ranges
            pref = np.clip(dif, 0, 1.0)
            pi[a, b] = (w * pref).sum()

    phi_pos = pi.sum(axis=1) / (n - 1)
    phi_neg = pi.sum(axis=0) / (n - 1)
    out = df.copy()
    out["promethee_phi"] = phi_pos - phi_neg
    return out.sort_values("promethee_phi", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    df, pasta = montar_matriz_decisao()

    print("=== Matriz de decisão ===")
    print(df[["eps"] + CRITERIOS].to_string(index=False))

    resumo = []
    for nome_cenario, ordem in CENARIOS.items():
        pesos, lam, CR = pesos_ahp(ordem)
        r_topsis = topsis(df, pesos)
        r_prom = promethee2(df, pesos)

        vt = int(r_topsis.iloc[0]["eps"])
        vp = int(r_prom.iloc[0]["eps"])

        print(f"\n=== Cenário: {nome_cenario} ===")
        print("Pesos:", {c: round(pesos[c], 3) for c in CRITERIOS}, f" (CR={CR:.4f})")
        print(f"TOPSIS -> eps={vt}   PROMETHEE II -> eps={vp}")

        resumo.append({"cenário": nome_cenario, "TOPSIS": vt, "PROMETHEE_II": vp})

        # guarda os resultados do cenário "robustez dominante" pra gerar a figura final
        if nome_cenario == "robustez dominante (atual)":
            r_topsis_final = r_topsis

    print("\n=== Resumo comparativo entre cenários ===")
    print(pd.DataFrame(resumo).to_string(index=False))

    # figura: fronteira com a solução final (cenário robustez dominante) destacada
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df["f1"], df["f2"], color="steelblue", s=60, zorder=2, label="Soluções avaliadas")
    escolhida = r_topsis_final.iloc[0]
    ax.scatter(escolhida["f1"], escolhida["f2"], color="firebrick", s=160,
               marker="*", zorder=3, label=f"Escolhida: {int(escolhida['eps'])} equipes")
    for _, row in df.iterrows():
        ax.annotate(f"{int(row['eps'])}", (row["f1"], row["f2"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("f1 — Distância total (km)", fontsize=15)
    ax.set_ylabel("f2 — Número de equipes", fontsize=15)
    ax.set_title("Fronteira avaliada na decisão multiatributo", fontsize=15)
    ax.legend(fontsize=15)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(pasta / "decisao_fronteira.png", dpi=150)
    print(f"\nFigura salva em: {pasta / 'decisao_fronteira.png'}")
    plt.show()