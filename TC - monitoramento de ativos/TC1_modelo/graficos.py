import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

if __name__ == "__main__":

    pasta = Path.cwd()
    arquivo = pasta / 'resultados_V1.csv'
    arquivo_cordenadas = pasta / 'identificadores.csv'

    dados_pareto = pd.read_csv(arquivo, sep=';', decimal=',')
    dados_corden = pd.read_csv(arquivo_cordenadas, sep = ';', decimal = ',')
    
    # Fonteira de Pareto

    fig, ax = plt.subplots(figsize=(8, 8))

    dados_pareto.plot(x='f1', y='f2',
        kind='scatter',
        ax=ax,
        marker='o',
        color='blue',
        #label='Soluções'
    )

    ax.set_xlabel('Distancia')
    ax.set_ylabel('Equipes empregadas')
    ax.set_title('Fronteira de Pareto')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

    # Arranjo coordenadas
    ativos = dados_corden[['lat_ativo', 'lon_ativo', 'ativo_id']].drop_duplicates()
    ativos['ativo_id'] = ativos['ativo_id']+1
    bases = dados_corden[['lat_base', 'lon_base', 'base_id']].drop_duplicates()
    bases['base_id'] = bases['base_id']+1

    fig, ax = plt.subplots(figsize=(8, 8))


    ativos.plot(x='lat_ativo', y='lon_ativo',
        kind='scatter',
        ax=ax,
        marker='o',
        color='blue',
        #label='ativos'
    )

    ax.set_xlabel('Lat')
    ax.set_ylabel('long')
    ax.set_title('map')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

  
    
    
