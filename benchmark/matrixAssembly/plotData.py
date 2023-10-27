#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
#%%
bench_data = pd.read_csv("results/benchmark.csv")
#%%
bench_data = bench_data[~bench_data["description"].str.contains("application::main")]
bench_data['nProcs'] = pd.to_numeric(bench_data['nProcs'])
bench_data['mesh'] = bench_data['mesh'].str.replace('hex_', '')
bench_data['mesh'] = pd.to_numeric(bench_data['mesh'])
bench_data['totalTime'] = pd.to_numeric(bench_data['totalTime'])
bench_data
#%%
bench_data
bench_data['nCells'] = bench_data['mesh']*bench_data['mesh']
filtered = bench_data[bench_data["proc"] == "processor0"]
filtered = filtered[filtered["description"].str.contains("Energy")]
filtered
# %%
# Draw a nested barplot by species and sex
sns.lineplot(
    data=filtered, x="nCells", y="totalTime", hue="description",style="nProcs"
)
plt.show()
