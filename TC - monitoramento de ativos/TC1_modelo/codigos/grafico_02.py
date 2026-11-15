"""
Gráfico de atribuição ativo -> base, no estilo da Figura do enunciado
(ex-resultado.png): bases como pentágono (vazio = disponível, preenchido
= ocupada), ativos como ponto, e uma linha ligando cada ativo à sua base.

Pré-requisitos (gerados pelo script do modelo, já com a correção do S
dinâmico aplicada):
  - identificadores.csv   (lat/lon de bases e ativos, com base_id/ativo_id)
  - atribuicoes_V2.csv    (formato longo: eps; ativo_id; base_id)
  - resultados_V2.csv     (um ponto da fronteira por linha, com a coluna 'bases')
"""
import ast
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


def plotar_atribuicao(eps_escolhido, pasta=None, salvar=True, mostrar=True):
    pasta = Path(pasta or Path.cwd())

    ident = pd.read_csv(pasta / "identificadores.csv", sep=";", decimal=",")
    bases_coord = ident.drop_duplicates("base_id")[["base_id", "lat_base", "lon_base"]].set_index("base_id")
    ativos_coord = ident.drop_duplicates("ativo_id")[["ativo_id", "lat_ativo", "lon_ativo"]].set_index("ativo_id")

    atrib = pd.read_csv(pasta / "atribuicoes_V2.csv", sep=";", decimal=",")
    atrib_eps = atrib[atrib["eps"] == eps_escolhido]
    if atrib_eps.empty:
        raise ValueError(f"Nenhuma atribuição encontrada para eps={eps_escolhido}. "
                          f"Valores disponíveis: {sorted(atrib['eps'].unique())}")

    pontos = pd.read_csv(pasta / "resultados_V2.csv", sep=";", decimal=",")
    linha = pontos[pontos["eps"] == eps_escolhido].iloc[0]
    # 'bases' foi salvo como texto de lista Python (ex.: "[0, 2]") -> parse seguro
    bases_ocupadas = set(ast.literal_eval(str(linha["bases"])))

    fig, ax = plt.subplots(figsize=(9, 9))

    for base_id, row in bases_coord.iterrows():
        ocupada = base_id in bases_ocupadas
        ax.scatter(row["lon_base"], row["lat_base"],
                   marker="p", s=260,
                   facecolor="tab:green" if ocupada else "white",
                   edgecolor="tab:green", linewidth=1.6, zorder=3)

    for _, linha_a in atrib_eps.iterrows():
        ativo_id, base_id = linha_a["ativo_id"], linha_a["base_id"]
        lat_a, lon_a = ativos_coord.loc[ativo_id, ["lat_ativo", "lon_ativo"]]
        lat_b, lon_b = bases_coord.loc[base_id, ["lat_base", "lon_base"]]
        ax.plot([lon_a, lon_b], [lat_a, lat_b], color="steelblue",
                linewidth=0.9, alpha=0.7, zorder=1)

    ax.scatter(ativos_coord["lon_ativo"], ativos_coord["lat_ativo"],
               marker="o", s=35, color="purple", zorder=2)

    legenda = [
        Line2D([0], [0], marker="p", color="w", markerfacecolor="white",
               markeredgecolor="tab:green", markersize=15, label="Bases disponível"),
        Line2D([0], [0], marker="p", color="w", markerfacecolor="tab:green",
               markeredgecolor="tab:green", markersize=15, label="Bases ocupada"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="purple",
               markersize=9, label="Ativos"),
    ]
    ax.legend(handles=legenda, loc="best")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Atribuição de ativos às bases (S = {eps_escolhido} equipes)")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if salvar:
        caminho = pasta / f"atribuicao_S{eps_escolhido}.png"
        plt.savefig(caminho, dpi=150)
        print(f"Figura salva em: {caminho}")
    if mostrar:
        plt.show()


if __name__ == "__main__":
    # troque pelo S (número de equipes) da solução que vocês escolherem
    # na tomada de decisão multiatributo (Entrega 3.v)
    plotar_atribuicao(eps_escolhido=2)