"""
Gráfico de atribuição ativo -> base (estilo ex-resultado.png), lendo direto
do resultados_modelo2.csv gerado pelo modelo_2.py.
"""
import ast
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


def plotar_atribuicao(eps_escolhido, pasta=None, salvar=True, mostrar=True):
    pasta = Path(pasta or Path.cwd())

    ident = pd.read_csv(pasta / "identificadores.csv", sep=";", decimal=",")

    bases_coord = ident.drop_duplicates("base_id")[["base_id", "lat_base", "lon_base"]].copy()
    bases_coord["base_id"] += 1   # alinha com a numeração 1..m do modelo
    bases_coord = bases_coord.set_index("base_id")

    ativos_coord = ident.drop_duplicates("ativo_id")[["ativo_id", "lat_ativo", "lon_ativo"]].copy()
    ativos_coord["ativo_id"] += 1   # alinha com a numeração 1..n do modelo
    ativos_coord = ativos_coord.set_index("ativo_id")

    pontos = pd.read_csv(pasta / "resultados_modelo2.csv", sep=";", decimal=",")
    linha = pontos[pontos["eps"] == eps_escolhido]
    if linha.empty:
        raise ValueError(f"Nenhum ponto encontrado para eps={eps_escolhido}. "
                          f"Valores disponíveis: {sorted(pontos['eps'].unique())}")
    linha = linha.iloc[0]

    bases_ocupadas = set(ast.literal_eval(str(linha["bases"])))
    atrib = ast.literal_eval(str(linha["atrib"]))              # {ativo_id: base_id}
    atrib = {int(k): int(v) for k, v in atrib.items()}

    fig, ax = plt.subplots(figsize=(9, 9))

    for base_id, row in bases_coord.iterrows():
        ocupada = base_id in bases_ocupadas
        ax.scatter(row["lon_base"], row["lat_base"],
                   marker="p", s=260,
                   facecolor="tab:green" if ocupada else "white",
                   edgecolor="tab:green", linewidth=1.6, zorder=3)

    for ativo_id, base_id in atrib.items():
        lat_a, lon_a = ativos_coord.loc[ativo_id, ["lat_ativo", "lon_ativo"]]
        lat_b, lon_b = bases_coord.loc[base_id, ["lat_base", "lon_base"]]
        ax.plot([lon_a, lon_b], [lat_a, lat_b], color="steelblue",
                linewidth=0.9, alpha=0.7, zorder=1)

    ax.scatter(ativos_coord["lon_ativo"], ativos_coord["lat_ativo"],
               marker="o", s=35, color="purple", zorder=2)

    legenda = [
        Line2D([0], [0], marker="p", color="w", markerfacecolor="white",
               markeredgecolor="tab:green", markersize=15, label="Base disponível"),
        Line2D([0], [0], marker="p", color="w", markerfacecolor="tab:green",
               markeredgecolor="tab:green", markersize=15, label="Base ocupada"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="purple",
               markersize=9, label="Ativos"),
    ]
    ax.legend(handles=legenda, loc="best")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Atribuição de ativos às bases ({eps_escolhido} equipes)")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if salvar:
        caminho = pasta / f"atribuicao_S{eps_escolhido}.png"
        plt.savefig(caminho, dpi=150)
        print(f"Figura salva em: {caminho}")
    if mostrar:
        plt.show()


if __name__ == "__main__":
    plotar_atribuicao(eps_escolhido=3)