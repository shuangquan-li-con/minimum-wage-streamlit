import streamlit as st
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Minimum Wage DiD Dashboard", layout="wide")

st.title("Minimum Wage and Employment: DiD Dashboard")
st.write("This dashboard estimates the effect of New Jersey's minimum wage increase on fast-food employment relative to Pennsylvania.")

df = pd.read_csv("njmin_clean.csv")

df["treated"] = df["state"]

df_before = df.copy()
df_before["post"] = 0
df_before["employment"] = df_before["fte_before"]
df_before["wage"] = df_before["wage_st"]

df_after = df.copy()
df_after["post"] = 1
df_after["employment"] = df_after["fte_after"]
df_after["wage"] = df_after["wage_st2"]

df_long = pd.concat([df_before, df_after], ignore_index=True)
df_long["did"] = df_long["treated"] * df_long["post"]

st.sidebar.header("What-if Scenario")
effect_multiplier = st.sidebar.slider(
    "Treatment effect multiplier",
    min_value=0.5,
    max_value=2.0,
    value=1.0,
    step=0.1
)

model = smf.ols("employment ~ treated + post + did", data=df_long).fit()

did_effect = model.params["did"]
ci_low, ci_high = model.conf_int().loc["did"]

adjusted_effect = did_effect * effect_multiplier
adjusted_ci_low = ci_low * effect_multiplier
adjusted_ci_high = ci_high * effect_multiplier

st.subheader("Estimated Policy Effect")

col1, col2, col3 = st.columns(3)

col1.metric("DiD Estimate", f"{did_effect:.2f} FTE workers")
col2.metric("95% CI Lower", f"{ci_low:.2f}")
col3.metric("95% CI Upper", f"{ci_high:.2f}")

st.write(
    f"If the treatment effect is multiplied by **{effect_multiplier:.1f}**, "
    f"the estimated effect becomes **{adjusted_effect:.2f} FTE workers** "
    f"(95% CI: [{adjusted_ci_low:.2f}, {adjusted_ci_high:.2f}])."
)

st.subheader("Employment Before and After Policy")

plot_data = df_long.groupby(["post", "treated"])["employment"].mean().reset_index()
plot_data["group"] = plot_data["treated"].map({0: "Pennsylvania", 1: "New Jersey"})
plot_data["period"] = plot_data["post"].map({0: "Before", 1: "After"})

fig, ax = plt.subplots()
for group in plot_data["group"].unique():
    temp = plot_data[plot_data["group"] == group]
    ax.plot(temp["period"], temp["employment"], marker="o", label=group)

ax.set_ylabel("Average FTE Employment")
ax.set_title("Average Employment Before and After Policy")
ax.legend()
st.pyplot(fig)

st.subheader("Wage Before and After Policy")

wage_data = df_long.groupby(["post", "treated"])["wage"].mean().reset_index()
wage_data["group"] = wage_data["treated"].map({0: "Pennsylvania", 1: "New Jersey"})
wage_data["period"] = wage_data["post"].map({0: "Before", 1: "After"})

fig2, ax2 = plt.subplots()
for group in wage_data["group"].unique():
    temp = wage_data[wage_data["group"] == group]
    ax2.plot(temp["period"], temp["wage"], marker="o", label=group)

ax2.set_ylabel("Average Starting Wage")
ax2.set_title("Average Wage Before and After Policy")
ax2.legend()
st.pyplot(fig2)

st.subheader("Interpretation")
st.write(
    "The Difference-in-Differences estimate suggests a positive effect of the minimum wage increase "
    "on employment, but the confidence interval includes zero. This means the result should be interpreted "
    "with caution. The dashboard also shows that New Jersey wages increased after the policy, supporting "
    "the idea that the treatment was actually implemented."
)