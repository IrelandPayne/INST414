import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

#reading in data 
tickers = ["VEA", "VGT", "VIG", "VO", "VOO", 
           "VTI", "VTV", "VUG", "VWO", "VXUS", 
           "VYM", "VT", "MGK", "VYMI", "VDE", 
           "VOOG", "VV", "VOX", "VNQ", "VB"]

all_dfs = []
for etf in tickers:
    df = pd.read_csv(f"{etf}_holdings.csv")
    df = df[["TICKER", "% OF FUNDS*", "SUB-INDUSTRY", "COUNTRY"]].copy()
    
#cleaning data per cosine comparison
    df["TICKER"] = df["TICKER"].astype(str).str.strip()
    df["SUB-INDUSTRY"] = df["SUB-INDUSTRY"].astype(str).str.strip().replace("---", "Other").fillna("Other")
    df["COUNTRY"] = df["COUNTRY"].astype(str).str.strip().replace("---", "Other").fillna("Other")
    
    df["% OF FUNDS*"] = (
        df["% OF FUNDS*"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace("<", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("---", "0", regex=False)
    )
    df["% OF FUNDS*"] = pd.to_numeric(df["% OF FUNDS*"]).fillna(0) / 100
    df["ETF"] = etf
    all_dfs.append(df)

#prepping data/assigning cosine 
master = pd.concat(all_dfs, ignore_index=True)

def get_sim_df(df, feature):
    pivot = df.pivot_table(
        index="ETF",
        columns=feature,
        values="% OF FUNDS*",
        aggfunc="sum",
        fill_value=0
    )
    pivot = pivot.div(pivot.sum(axis=1), axis=0) 
    sim = cosine_similarity(pivot)
    return pd.DataFrame(sim, index=pivot.index, columns=pivot.index)

#different cosine comparisons 
sim_ticker = get_sim_df(master, "TICKER")
sim_industry = get_sim_df(master, "SUB-INDUSTRY")
sim_country = get_sim_df(master, "COUNTRY")
sim_overall = (sim_ticker + sim_industry + sim_country) / 3

metrics = {
    "Ticker": sim_ticker, 
    "Sub-Industry": sim_industry, 
    "Country": sim_country,
    "Overall": sim_overall
}
#queries
targets = ["VOO", "VGT", "VXUS"]

#print statements/heat maps
for target in targets:
    fig, axes = plt.subplots(1, 4, figsize=(20, 8))
    
    for i, (name, df_sim) in enumerate(metrics.items()):
        top_10 = df_sim[target].drop(target).sort_values(ascending=False).head(10)
        print(f"\nTop 10 {name} similarity to {target}:")
        print(top_10.round(4))
        
        sns.heatmap(
            df_sim[[target]].sort_values(by=target, ascending=False),
            annot=True,
            ax=axes[i],
            cmap="coolwarm",
            fmt=".4f"
        )
        axes[i].set_title(f"{name} Similarity")
        
    plt.tight_layout()
    plt.show()