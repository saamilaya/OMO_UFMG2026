"""
Visualizacao dos indicadores de qualidade (Entrega 3.iv): Delta (espacamento
real vs. ideal) e Hipervolume (area dominada), sobre a fronteira exata do
AUGMECON2.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path


def _carregar_fronteira(pasta=None):
    pasta = Path(pasta or Path.cwd())
    df = pd.read_csv(pasta / "resultados_modelo2.csv", sep=";", decimal=",")
    df = df.sort_values("f2").reset_index(drop=True)

    f1 = df["f1"].to_numpy(dtype=float)
    f2 = df["f2"].to_numpy(dtype=float)

    u = np.array([f1.min(), f2.min()])
    nadir = np.array([f1.max(), f2.max()])
    rango = nadir - u
    rango[rango == 0] = 1.0

    f1n = (f1 - u[0]) / rango[0]
    f2n = (f2 - u[1]) / rango[1]
    return f2, f1n, f2n


def grafico_delta(pasta=None, salvar=True, mostrar=True):
    eps, f1n, f2n = _carregar_fronteira(pasta)
    pasta = Path(pasta or Path.cwd())

    pts = np.column_stack([f1n, f2n])
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    d_media = d.mean()
    rotulos = [f"{int(eps[i])}\u2192{int(eps[i+1])}" for i in range(len(eps) - 1)]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(d))
    ax.bar(x, d, color="steelblue", width=0.6, label="Distância real entre vizinhos (d$_i$)")
    ax.axhline(d_media, color="firebrick", linestyle="--", linewidth=1.6,
               label=f"Espaçamento ideal / médio (d̄ = {d_media:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(rotulos)
    ax.set_xlabel("Par de soluções vizinhas (número de equipes)", fontsize=15)
    ax.set_ylabel("Distância euclidiana normalizada", fontsize=15)
    ax.set_title("Indicador Δ: espaçamento real da fronteira vs. ideal uniforme", fontsize=15)
    ax.legend(fontsize=15)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()

    if salvar:
        caminho = pasta / "delta_espacamento.png"
        plt.savefig(caminho, dpi=150)
        print(f"Figura salva em: {caminho}")
    if mostrar:
        plt.show()


def grafico_hipervolume(pasta=None, salvar=True, mostrar=True):
    eps, f1n, f2n = _carregar_fronteira(pasta)
    pasta = Path(pasta or Path.cwd())

    pts = np.column_stack([f1n, f2n])
    ordem = np.argsort(pts[:, 0])
    pts = pts[ordem]
    r = np.array([1.1, 1.1])

    fig, ax = plt.subplots(figsize=(7, 7))

    hv_total = 0.0
    for i in range(len(pts)):
        xi, yi = pts[i]
        x_prox = pts[i + 1, 0] if i + 1 < len(pts) else r[0]
        largura = x_prox - xi
        altura = r[1] - yi
        hv_total += largura * altura
        ax.add_patch(Rectangle((xi, yi), largura, altura,
                                facecolor="steelblue", alpha=0.25,
                                edgecolor="none", zorder=1))

    ax.step(pts[:, 0], pts[:, 1], where="post", color="steelblue", linewidth=1.8, zorder=2)
    ax.scatter(pts[:, 0], pts[:, 1], color="steelblue", s=50, zorder=3, label="Fronteira obtida")
    ax.scatter(*r, color="firebrick", marker="X", s=90, zorder=3,
               label="Ponto de referência (1,1 × nadir)")

    ax.set_xlim(-0.05, r[0] + 0.05)
    ax.set_ylim(-0.05, r[1] + 0.05)
    ax.set_xlabel("f1 normalizado (distância)", fontsize=15)
    ax.set_ylabel("f2 normalizado (equipes)", fontsize=15)
    ax.set_title(f"Área dominada (Hipervolume ≈ {hv_total:.4f})", fontsize=15)
    ax.legend(loc="lower left", fontsize=15)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if salvar:
        caminho = pasta / "hipervolume_area.png"
        plt.savefig(caminho, dpi=150)
        print(f"Figura salva em: {caminho}")
    if mostrar:
        plt.show()


if __name__ == "__main__":
    grafico_delta()
    grafico_hipervolume()