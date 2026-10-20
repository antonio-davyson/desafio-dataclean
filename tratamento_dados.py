"""
DataClean Co. — Tratamento de dados de clientes/vendas
Desafio Rocketseat

Carrega dados_clientes.csv, trata ausentes, duplicatas e outliers,
aplica normalização/encoding, padroniza datas, gera visualizações e
exporta o dataset limpo + um relatório antes/depois em .xlsx.

Como executar:
    python tratamento_dados.py

Requisitos:
    pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")  # gera os gráficos como arquivo, sem precisar de tela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

ARQUIVO_ENTRADA = "dados_clientes.csv"
PASTA_SAIDA = "saida"
os.makedirs(PASTA_SAIDA, exist_ok=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)


def secao(titulo: str) -> None:
    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1) Carregar os dados + informações básicas
# ---------------------------------------------------------------------------

secao("1) CARREGANDO OS DADOS")
df = pd.read_csv(ARQUIVO_ENTRADA)

print(f"\nArquivo carregado: {ARQUIVO_ENTRADA}")
print(f"Total de registros: {df.shape[0]}")
print(f"Total de colunas:   {df.shape[1]}")

print("\nPrimeiras 5 linhas:")
print(df.head())

print("\nTipos de dado por coluna:")
print(df.dtypes)

# Guarda uma cópia do estado "bruto" para o relatório antes/depois
df_bruto = df.copy()


# ---------------------------------------------------------------------------
# 2) Tratar dados ausentes
# ---------------------------------------------------------------------------

secao("2) TRATANDO DADOS AUSENTES")

print("\nValores ausentes por coluna (antes):")
print(df.isna().sum())

# idade / renda_anual / valor_compra: ausência provavelmente aleatória
# (poucas linhas, sem padrão aparente) -> preenchemos com a MEDIANA de cada
# coluna. Mediana em vez de média porque essas colunas têm outliers fortes
# (ex: renda de R$750.000 no meio de rendas de ~R$100-190 mil), e a média
# seria puxada por eles; a mediana não.
for coluna in ["idade", "renda_anual", "valor_compra"]:
    mediana = df[coluna].median()
    qtd_ausentes = df[coluna].isna().sum()
    df[coluna] = df[coluna].fillna(mediana)
    print(f"  '{coluna}': {qtd_ausentes} valor(es) ausente(s) preenchido(s) com a mediana ({mediana:.2f})")

# data_compra: não faz sentido "inventar" uma data de compra -- não existe
# estatística razoável pra isso (diferente de idade/renda). Como as colunas
# derivadas de data (ano, mês, dia da semana) dependem de uma data real,
# optamos por REMOVER as linhas sem data em vez de preencher com um valor
# fictício que distorceria essas análises.
qtd_sem_data = df["data_compra"].isna().sum()
df = df.dropna(subset=["data_compra"]).reset_index(drop=True)
print(f"  'data_compra': {qtd_sem_data} linha(s) removida(s) por não terem data (não é seguro inventar uma data de compra)")

print("\nValores ausentes por coluna (depois):")
print(df.isna().sum())
print(f"\nTotal de linhas após tratar ausentes: {len(df)} (era {len(df_bruto)})")


# ---------------------------------------------------------------------------
# 3) Detectar e tratar duplicatas
# ---------------------------------------------------------------------------

secao("3) DETECTANDO E TRATANDO DUPLICATAS")

# 3a) Duplicatas de linha inteira (o caso "clássico": a mesma linha, idêntica,
# aparece mais de uma vez)
duplicatas_totais = df.duplicated(keep=False)
print(f"\nLinhas 100% idênticas (todas as colunas iguais): {duplicatas_totais.sum()}")

if duplicatas_totais.sum() > 0:
    # Mantém a linha com menos valores ausentes (a "mais completa"/relevante);
    # em caso de empate, mantém a primeira ocorrência.
    df["_qtd_nulos"] = df.isna().sum(axis=1)
    df = df.sort_values("_qtd_nulos").drop_duplicates(
        subset=[c for c in df.columns if c != "_qtd_nulos"], keep="first"
    )
    df = df.drop(columns="_qtd_nulos").sort_index().reset_index(drop=True)
    print("Duplicatas de linha inteira removidas, mantendo a versão mais completa de cada uma.")
else:
    print("Nenhuma linha 100% duplicada encontrada.")

# 3b) id_cliente duplicado (id_cliente deveria ser único, mas não é --
# isso é um problema de qualidade diferente de "linha duplicada": são
# clientes DIFERENTES (nomes diferentes) compartilhando o mesmo código,
# não a mesma pessoa registrada duas vezes. Por isso NÃO removemos essas
# linhas -- apagar uma delas destruiria o dado de um cliente real. Só
# reportamos o problema, que na vida real voltaria pro time de origem
# do dado para correção do identificador.
ids_duplicados = df["id_cliente"][df["id_cliente"].duplicated(keep=False)]
qtd_ids_duplicados = ids_duplicados.nunique()
print(
    f"\nAviso de qualidade (não removido automaticamente): {qtd_ids_duplicados} "
    f"código(s) de id_cliente aparecem em mais de um registro, mas com nomes "
    f"diferentes -- indica clientes distintos com o mesmo código, não uma "
    f"duplicata real. Recomenda-se reportar ao time de origem do dado."
)


# ---------------------------------------------------------------------------
# 4) Analisar outliers
# ---------------------------------------------------------------------------

secao("4) ANALISANDO OUTLIERS (renda_anual e valor_compra)")

colunas_outlier = ["renda_anual", "valor_compra"]


def limites_iqr(serie: pd.Series) -> tuple[float, float]:
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


for coluna in colunas_outlier:
    limite_inf, limite_sup = limites_iqr(df[coluna])
    outliers = df[(df[coluna] < limite_inf) | (df[coluna] > limite_sup)]
    print(f"\n'{coluna}': limites aceitáveis (IQR) = [{limite_inf:.2f}, {limite_sup:.2f}]")
    print(f"  {len(outliers)} outlier(s) encontrado(s): {outliers[coluna].tolist()}")

    # Boxplot ANTES do tratamento
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[coluna])
    plt.title(f"Boxplot de {coluna} (antes do tratamento de outliers)")
    plt.tight_layout()
    plt.savefig(f"{PASTA_SAIDA}/boxplot_{coluna}_antes.png")
    plt.close()

    # Decisão: CAPPING (winsorização) em vez de remover ou substituir por
    # mediana. Justificativa: valores altos de renda/compra são plausíveis
    # de existir de verdade (um cliente pode legitimamente ganhar/gastar
    # muito mais que a média) -- excluir a linha jogaria fora um cliente
    # real. Mas mantê-los sem tratamento distorceria a normalização (etapa
    # 5), então limitamos (capping) ao limite do IQR, preservando a
    # informação de "é um valor alto" sem deixar o valor extremo dominar
    # a escala.
    df[coluna] = df[coluna].clip(lower=limite_inf, upper=limite_sup)

    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[coluna])
    plt.title(f"Boxplot de {coluna} (depois do capping)")
    plt.tight_layout()
    plt.savefig(f"{PASTA_SAIDA}/boxplot_{coluna}_depois.png")
    plt.close()

print(f"\nGráficos de boxplot (antes/depois) salvos em '{PASTA_SAIDA}/'.")


# ---------------------------------------------------------------------------
# 5) Trabalhar com datas
# ---------------------------------------------------------------------------

secao("5) PADRONIZANDO DATAS")

print("\nFormatos de data encontrados na coluna original:", df["data_compra"].unique())

# A coluna vem com formatos misturados: '2024-01-15' (ano-mês-dia com
# hífen), '2024/03/10' (ano-mês-dia com barra) e '15/02/2024'
# (dia/mês/ano). Usar format="mixed" com dayfirst=True parece a solução
# óbvia, mas TEM UM BUG: o dayfirst acaba trocando mês por dia mesmo nos
# formatos ISO (ano primeiro), onde essa opção nem deveria se aplicar --
# '2024-04-05' virava '2024-05-04' silenciosamente, sem erro nenhum.
# Por isso, em vez de deixar o pandas "adivinhar", tentamos formatos
# explícitos, um de cada vez, na ordem em que aparecem no dado real:
def parsear_data(valor):
    for formato in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y"):
        try:
            return pd.to_datetime(valor, format=formato)
        except (ValueError, TypeError):
            continue
    return pd.NaT


df["data_compra"] = df["data_compra"].apply(parsear_data)

df["ano_compra"] = df["data_compra"].dt.year
df["mes_compra"] = df["data_compra"].dt.month
df["dia_semana_compra"] = df["data_compra"].dt.day_name()

print("\nExemplo após padronização:")
print(df[["data_compra", "ano_compra", "mes_compra", "dia_semana_compra"]].head())


# ---------------------------------------------------------------------------
# 6) Normalização e encoding
# ---------------------------------------------------------------------------

secao("6) NORMALIZAÇÃO E ENCODING")

# Normalização (Min-Max, escala 0-1) das colunas numéricas contínuas.
# Cria colunas novas com sufixo _norm, preservando os valores originais
# (em reais/anos) para leitura e para os gráficos.
colunas_numericas = ["idade", "renda_anual", "valor_compra"]
scaler = MinMaxScaler()
df[[f"{c}_norm" for c in colunas_numericas]] = scaler.fit_transform(df[colunas_numericas])

print("\nColunas numéricas normalizadas (0 a 1):", [f"{c}_norm" for c in colunas_numericas])
print(df[[f"{c}_norm" for c in colunas_numericas]].describe().loc[["min", "max"]])

# Encoding das colunas categóricas nominais (sem ordem natural entre as
# categorias) -> one-hot encoding. Mantém as colunas de texto originais
# (mais legíveis pro relatório) e acrescenta as colunas 0/1.
colunas_categoricas = ["genero", "categoria_produto", "regiao"]
df_encoded = pd.get_dummies(df, columns=colunas_categoricas, prefix=colunas_categoricas)

print(f"\nColunas antes do encoding: {len(df.columns)}")
print(f"Colunas depois do encoding (one-hot): {len(df_encoded.columns)}")


# ---------------------------------------------------------------------------
# 7) Visualizações (desafio extra)
# ---------------------------------------------------------------------------

secao("7) GERANDO VISUALIZAÇÕES")

plt.figure(figsize=(7, 5))
sns.histplot(df["renda_anual"], kde=True)
plt.title("Distribuição da renda anual dos clientes")
plt.xlabel("Renda anual (R$)")
plt.tight_layout()
plt.savefig(f"{PASTA_SAIDA}/histograma_renda.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="categoria_produto", y="valor_compra")
plt.title("Gasto por categoria de produto")
plt.xlabel("Categoria")
plt.ylabel("Valor da compra (R$)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{PASTA_SAIDA}/boxplot_gasto_por_categoria.png")
plt.close()

plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="idade", y="valor_compra", hue="genero")
plt.title("Relação entre idade e valor gasto")
plt.tight_layout()
plt.savefig(f"{PASTA_SAIDA}/scatter_idade_valor.png")
plt.close()

plt.figure(figsize=(6, 5))
correlacao = df[["idade", "renda_anual", "valor_compra"]].corr()
sns.heatmap(correlacao, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlação entre variáveis numéricas")
plt.tight_layout()
plt.savefig(f"{PASTA_SAIDA}/heatmap_correlacao.png")
plt.close()

print(f"4 gráficos salvos em '{PASTA_SAIDA}/': histograma_renda.png, "
      f"boxplot_gasto_por_categoria.png, scatter_idade_valor.png, heatmap_correlacao.png")


# ---------------------------------------------------------------------------
# 8) Exportar dataset limpo + relatório antes/depois
# ---------------------------------------------------------------------------

secao("8) EXPORTANDO RESULTADOS")

caminho_csv = f"{PASTA_SAIDA}/dados_clientes_limpo.csv"
df_encoded.to_csv(caminho_csv, index=False, encoding="utf-8")
print(f"\nDataset limpo exportado para: {caminho_csv}")

# Relatório antes/depois em .xlsx
resumo_antes = pd.DataFrame({
    "métrica": ["total_linhas", "total_colunas", "ausentes_idade", "ausentes_renda_anual",
                "ausentes_valor_compra", "ausentes_data_compra", "linhas_duplicadas_100pct"],
    "valor": [
        len(df_bruto), df_bruto.shape[1],
        df_bruto["idade"].isna().sum(), df_bruto["renda_anual"].isna().sum(),
        df_bruto["valor_compra"].isna().sum(), df_bruto["data_compra"].isna().sum(),
        df_bruto.duplicated().sum(),
    ],
})

resumo_depois = pd.DataFrame({
    "métrica": ["total_linhas", "total_colunas", "ausentes_idade", "ausentes_renda_anual",
                "ausentes_valor_compra", "ausentes_data_compra", "linhas_duplicadas_100pct"],
    "valor": [
        len(df), len(df.columns),
        df["idade"].isna().sum(), df["renda_anual"].isna().sum(),
        df["valor_compra"].isna().sum(), df["data_compra"].isna().sum(),
        df.duplicated().sum(),
    ],
})

caminho_xlsx = f"{PASTA_SAIDA}/relatorio_antes_depois.xlsx"
with pd.ExcelWriter(caminho_xlsx, engine="openpyxl") as writer:
    resumo_antes.to_excel(writer, sheet_name="Resumo Antes", index=False)
    resumo_depois.to_excel(writer, sheet_name="Resumo Depois", index=False)
    df_bruto.describe().to_excel(writer, sheet_name="Estatísticas Antes")
    df.describe().to_excel(writer, sheet_name="Estatísticas Depois")

print(f"Relatório antes/depois exportado para: {caminho_xlsx}")

secao("CONCLUÍDO")
print(f"\nTudo pronto! Veja a pasta '{PASTA_SAIDA}/' para o CSV limpo, os gráficos e o relatório.")
