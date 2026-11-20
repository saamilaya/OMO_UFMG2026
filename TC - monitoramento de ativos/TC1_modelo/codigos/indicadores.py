"""
Indicadores de qualidade da fronteira (Entrega 3.iv): Delta (spread) e
Hipervolume, calculados uma unica vez sobre a fronteira exata do AUGMECON2.
"""
import numpy as np
import pandas as pd
from pathlib import Path


def calcular_indicadores(pasta=None):
    pasta = Path(pasta or Path.cwd())
    df = pd.read_csv(pasta / "resultados_modelo2.csv", sep=";", decimal=",")
    df = df.sort_values("f2").reset_index(drop=True)

    f1 = df["f1"].to_numpy(dtype=float)
    f2 = df["f2"].to_numpy(dtype=float)

    # ponto utopico (melhor valor de cada objetivo) e anti-utopico/nadir (pior)
    u = np.array([f1.min(), f2.min()])
    nadir = np.array([f1.max(), f2.max()])
    rango = nadir - u
    rango[rango == 0] = 1.0

    # normaliza os dois objetivos para [0,1], usando utopia/nadir como escala
    f1n = (f1 - u[0]) / rango[0]
    f2n = (f2 - u[1]) / rango[1]
    pts = np.column_stack([f1n, f2n])

    # ---------- Delta (spread/diversidade) ----------
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)   # distancias entre vizinhos
    d_media = d.mean()

    # distancia, em cada objetivo, entre o extremo da fronteira e a utopia
    # (como u foi tirado da propria fronteira exata, da exatamente 0 aqui)
    d_f1e = abs(pts[np.argmin(f1)][0])
    d_f2e = abs(pts[np.argmin(f2)][1])
    soma_extremos = d_f1e + d_f2e

    soma_irregularidade = np.abs(d - d_media).sum()
    delta = (soma_extremos + soma_irregularidade) / (soma_extremos + len(d) * d_media)

    # ---------- Hipervolume (S-metric) ----------
    r = np.array([1.1, 1.1])   # 1.1 x o anti-utopico, ja normalizado = (1.1, 1.1)
    ordem = np.argsort(pts[:, 0])
    ptsx = pts[ordem]

    hv = 0.0
    for i in range(len(ptsx)):
        xi, yi = ptsx[i]
        x_prox = ptsx[i + 1, 0] if i + 1 < len(ptsx) else r[0]
        hv += (x_prox - xi) * (r[1] - yi)

    return {"utopia": u, "nadir": nadir, "delta": delta, "hipervolume": hv}


if __name__ == "__main__":
    res = calcular_indicadores()
    print(f"Ponto utópico   (f1, f2): {res['utopia']}")
    print(f"Ponto anti-utópico/nadir: {res['nadir']}")
    print(f"Delta (spread):           {res['delta']:.4f}")
    print(f"Hipervolume (normalizado):{res['hipervolume']:.4f}")