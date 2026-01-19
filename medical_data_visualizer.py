import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import the data
df = pd.read_csv("medical_examination.csv")

# 2. Add overweight column
# BMI = weight (kg) / height (m)^2
df["overweight"] = (df["weight"] / ((df["height"] / 100) ** 2) > 25).astype(int)

# 3. Normalize data so that 0 is good and 1 is bad
df["cholesterol"] = df["cholesterol"].apply(lambda x: 0 if x == 1 else 1)
df["gluc"] = df["gluc"].apply(lambda x: 0 if x == 1 else 1)


def draw_cat_plot():
    # 4. Draw the Categorical Plot

    # 5. Create DataFrame for categorical plot using pd.melt
    df_cat = pd.melt(
        df,
        id_vars=["cardio"],
        value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"]
    )

    # 6. Group and reformat the data to show counts
    df_cat = (
        df_cat
        .groupby(["cardio", "variable", "value"])
        .size()
        .reset_index(name="total")
    )

    # 7. Convert data into long format and create chart using sns.catplot
    fig = sns.catplot(
        data=df_cat,
        x="variable",
        y="total",
        hue="value",
        col="cardio",
        kind="bar"
    ).fig

    # 8. Get the figure for the output
    return fig


def draw_heat_map():
    # 10. Draw the Heat Map

    # 11. Clean the data
    df_heat = df[
        (df["ap_lo"] <= df["ap_hi"]) &
        (df["height"] >= df["height"].quantile(0.025)) &
        (df["height"] <= df["height"].quantile(0.975)) &
        (df["weight"] >= df["weight"].quantile(0.025)) &
        (df["weight"] <= df["weight"].quantile(0.975))
    ]

    # 12. Calculate the correlation matrix
    corr = df_heat.corr()

    # 13. Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15. Plot the correlation matrix using sns.heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        square=True,
        linewidths=0.5,
        center=0,
        cbar_kws={"shrink": 0.5}
    )

    # 16. Do not modify the next two lines
    return fig
