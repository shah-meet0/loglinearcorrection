import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Style preamble
plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.usetex': True,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})

# Softer colors
COLOR_SIG = '#c44e52'      # muted red
COLOR_NONSIG = '#4c4c4c'   # dark gray


# Load file
filepath = r"./paper/data/replication_input.csv"
df = pd.read_csv(filepath)
df.columns

columns = ['variable_type', 'elasticity', 'elasticity_se', 'ols_coef', 'ppml_coef',
       'diff_elast_minus_ols', 'se_diff_elast_minus_ols',
       'test_elast_vs_ols_stat', 'test_elast_vs_ols_p',
       'diff_elast_minus_ppml', 'se_diff_elast_minus_ppml',
       'test_elast_vs_ppml_stat', 'test_elast_vs_ppml_p', 'nobs', 'time',
       'paper', 'panel']

df = df.query('paper <= 155 and paper != 117')

df['ols_diff_sig'] = df['test_elast_vs_ols_p'] < 0.1
df['ppml_diff_sig'] = df['test_elast_vs_ppml_p'] < 0.1

df['ols_diff_sig_5'] = df['test_elast_vs_ols_p'] < 0.05
df['ppml_diff_sig_5'] = df['test_elast_vs_ppml_p'] < 0.05
df['ols_ppml_diff_sig_5'] = df['test_ols_vs_ppml_p'] < 0.05

df['percent_diff_ols'] = (df['diff_elast_minus_ols'] / df['ols_coef']).abs() * 100

df['percent_diff_ols'].median()

# More nice columns

df['ols_sign_reversed'] = (np.sign(df['elasticity']) != np.sign(df['ols_coef'])) * df['ols_diff_sig_5']
df['ppml_sign_reversed'] = (np.sign(df['elasticity']) != np.sign(df['ppml_coef'])) * df['ppml_diff_sig_5']
df['ols_size_increase'] = ((df['elasticity'].abs() > df['ols_coef'].abs()) & (df['ols_sign_reversed'] ==0)) * df['ols_diff_sig_5']
df['ppml_size_increase'] = ((df['elasticity'].abs() > df['ppml_coef'].abs() )& (df['ppml_sign_reversed'] ==0)) * df['ppml_diff_sig_5']

df['ols_size_decrease'] = ((df['elasticity'].abs() < df['ols_coef'].abs()) & (df['ols_sign_reversed'] ==0)) * df['ols_diff_sig_5']
df['ppml_size_decrease'] = ((df['elasticity'].abs() < df['ppml_coef'].abs()) & (df['ppml_sign_reversed'] ==0)) * df['ppml_diff_sig_5']

#
counts = {
    'Sign reversed': df['ols_sign_reversed'].sum(),
    'Effect size decrease': df['ols_size_decrease'].sum(),
    'Effect size increase': df['ols_size_increase'].sum(),
       'Significantly different': df['ols_diff_sig_5'].sum(),
    'No change': (~df['ols_diff_sig_5']).sum(),   # not sig difference implies no change
}

counts_ppml = {
    'Sign reversed': df['ppml_sign_reversed'].sum(),
    'Effect size decrease': df['ppml_size_decrease'].sum(),
    'Effect size increase': df['ppml_size_increase'].sum(),
    'Significantly different': df['ppml_diff_sig_5'].sum(),
    'No change': (~df['ppml_diff_sig_5']).sum(),
}


def counts_to_markdown_table(counts_ols: dict, counts_ppml: dict) -> str:
    """
    Generate a markdown table comparing OLS and PPML counts.
    
    Columns: No Change | Significantly Different | Effect Increase | Effect Decrease | Sign Change
    Rows: OLS, PPML
    """
    headers = ['', 'No Change', 'Sig. Different', 'Effect Increase', 'Effect Decrease', 'Sign Change']
    
    ols_row = [
        'OLS',
        counts_ols['No change'],
        counts_ols['Significantly different'],
        counts_ols['Effect size increase'],
        counts_ols['Effect size decrease'],
        counts_ols['Sign reversed'],
    ]
    
    ppml_row = [
        'PPML',
        counts_ppml['No change'],
        counts_ppml['Significantly different'],
        counts_ppml['Effect size increase'],
        counts_ppml['Effect size decrease'],
        counts_ppml['Sign reversed'],
    ]
    
    # Build markdown table
    header_line = '| ' + ' | '.join(str(h) for h in headers) + ' |'
    separator = '| ' + ' | '.join(['---'] * len(headers)) + ' |'
    ols_line = '| ' + ' | '.join(str(v) for v in ols_row) + ' |'
    ppml_line = '| ' + ' | '.join(str(v) for v in ppml_row) + ' |'
    
    table = '\n'.join([header_line, separator, ols_line, ppml_line])
    return table


# Generate and print the markdown table
markdown_table = counts_to_markdown_table(counts, counts_ppml)
from pathlib import Path

path = Path("./paper/outputs/table_differences.md")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(markdown_table, encoding="utf-8")


print(markdown_table)

# Convert to series for plotting
counts = pd.Series(counts)
counts_ppml = pd.Series(counts_ppml)

#
# # Plot
# plt.figure(figsize=(10,6))
# counts.plot(kind='bar', color=['maroon', 'maroon', 'teal', 'teal', 'darkgray'])
#
# plt.ylabel('Count')
# plt.title('Classification of Elasticity vs OLS Coefficient Differences')
#
# # Add labels above bars
# for i, v in enumerate(counts):
#     plt.text(i, v + 0.5, str(v), ha='center')
#
# plt.xticks(rotation=360)
# plt.ylim(0, counts.max() +5)
# plt.savefig(r"C:\Users\Meet Shah\Desktop\Applied Micro Pres\Figures\ols_summ.pdf", dpi=1000, transparent=True)

# Summary statistics
#
# n_papers = len(df['paper'].unique())
# n_results = len(df['paper'])
#
# n_sig_diff_ols = df['ols_diff_sig'].sum()
# n_sig_diff_5_ols = df['ols_diff_sig_5'].sum()
#
# n_sig_diff_ppml = df['ppml_diff_sig'].sum()
# n_sig_diff_5_ppml = df['ppml_diff_sig_5'].sum()
#
#
# df_plot_1 = df[['elasticity', 'diff_elast_minus_ols', 'elasticity_se','ols_coef','se_diff_elast_minus_ols', 'test_elast_vs_ols_p', 'ols_diff_sig_5', 'nobs', 'variable_type','paper', 'panel']].copy()
# df_plot_1['abs_ols_coef'] = df_plot_1['ols_coef'].abs()
# df_plot_1['sign_change'] = (np.sign(df_plot_1['elasticity']) != np.sign(df_plot_1['ols_coef']))
# df_plot_1 = df_plot_1.query('abs(elasticity) < 1 and abs(ols_coef) < 1')
# df_plot_1.sort_values(by='diff_elast_minus_ols', inplace=True, ascending=True)
# df_plot_1.reset_index(drop=True, inplace=True)
# scale = 1
# df_plot_1['index'] =  scale * (df_plot_1.index + 1)
# #
# sig_color = {True: 'red', False: 'green'}
#
# # Define markers for sign_change
# marker_map = {1: '*', 0: 'o'}   # adjust shape as desired
#
# fig1, ax1 = plt.subplots(figsize=(9,8))
#
# # scatter object returned by seaborn
# import matplotlib.collections as mcoll
# from matplotlib.lines import Line2D
#
# # --- plot (turn off seaborn's auto legend) ---
# scatter = sns.scatterplot(
#     data=df_plot_1,
#     y='index',
#     x='diff_elast_minus_ols',
#     hue='ols_diff_sig_5',
#     style='sign_change',
#     markers={1: '*', -1: 'o', 0: 'o'},
#     size='abs_ols_coef',
#     sizes=(20, 100),
#     palette={True: 'red', False: 'black'},
#     ax=ax1,
#     legend=False,
#     zorder = 2,
# )
#
# # --- error bars (unchanged) ---
# for _, r in df_plot_1.iterrows():
#     ax1.errorbar(
#         r['diff_elast_minus_ols'], r['index'],
#         xerr=1.96 * r['se_diff_elast_minus_ols'],
#         fmt='none',
#         ecolor={True: 'red', False: 'black'}[r['ols_diff_sig_5']],
#         capsize=5,
#         zorder= 1,
#         alpha = 0.7
#     )
#
# # --- get the actual PathCollection for correct size legend ---
# points = next(c for c in ax1.collections if isinstance(c, mcoll.PathCollection))
# size_handles, size_labels = points.legend_elements(prop="sizes", alpha=1, num=4)
# size_labels = [0, 0.15, 0.30, 0.45, 0.60]
#
# # --- build color + shape legend (clean labels) ---
# sig_color = {True: 'red', False: 'green'}
# color_handles = [
#     Line2D([0], [0], marker='o', linestyle='none', color='none',
#            markerfacecolor=sig_color[True],  markeredgecolor='white', markeredgewidth=0.6,
#            label='Significant (5%)'),
# ]
# shape_handles = [
#     Line2D([0], [0], marker='*', linestyle='none', color='black', label='Sign Changed', markersize=10),
# ]
#
# # --- place legends: main bottom-right, size just above it (also bottom-right) ---
# leg1 = ax1.legend(color_handles + shape_handles,
#                   [h.get_label() for h in color_handles + shape_handles],
#                   title=None, frameon=False, loc='upper left', bbox_to_anchor=(-0.1, 1), ncol=1)
#
# leg2 = ax1.legend(size_handles, size_labels,
#                   title='Absolute value of OLS Coefficient', frameon=False,
#                   loc='upper left', bbox_to_anchor=(-0.1, 0.9), ncol=3)  # adjust 0.18 if overlap
#
# ax1.add_artist(leg1)  # keep both legends visible
# ax1.axvline(x=0, color='black', linestyle='--')
# ax1.get_yaxis().set_visible(False)
# ax1.set_frame_on(False)
# ax1.set_xlabel('')
# fig1.suptitle('Difference between DR Elasticity and OLS Coefficient with 95% CI', weight='bold', fontsize=16)
# ax1.spines['bottom'].set_visible(True)
# ax1.set_xlim(-0.6, 0.6)
# plt.tight_layout()
# plt.savefig(r"C:\Users\Meet Shah\Desktop\Applied Micro Pres\Figures\diff_graph.pdf", dpi=1000, transparent=True)

# FIG2
#
# df_plot_1 = df[['elasticity', 'diff_elast_minus_ppml', 'ppml_coef','se_diff_elast_minus_ppml', 'test_elast_vs_ols_p', 'ppml_diff_sig_5', 'nobs', 'variable_type']].copy()
# df_plot_1['abs_ppml_coef'] = df_plot_1['ppml_coef'].abs()
# df_plot_1['sign_change'] = (np.sign(df_plot_1['elasticity']) != np.sign(df_plot_1['ppml_coef']))
# df_plot_1 = df_plot_1.query('abs(elasticity) < 1 and abs(ppml_coef) < 1')
# df_plot_1.sort_values(by='diff_elast_minus_ppml', inplace=True, ascending=True)
# df_plot_1.reset_index(drop=True, inplace=True)
# scale = 1
# df_plot_1['index'] =  scale * (df_plot_1.index + 1)
#
# sig_color = {True: 'red', False: 'green'}
#
# # Define markers for sign_change
# marker_map = {1: '*', 0: 'o'}   # adjust shape as desired
#
# fig1, ax1 = plt.subplots(figsize=(9,8))
#
# # scatter object returned by seaborn
# import matplotlib.collections as mcoll
# from matplotlib.lines import Line2D
#
# # --- plot (turn off seaborn's auto legend) ---
# scatter = sns.scatterplot(
#     data=df_plot_1,
#     y='index',
#     x='diff_elast_minus_ppml',
#     hue='ppml_diff_sig_5',
#     style='sign_change',
#     markers={1: '*', 0: 'o'},
#     size='abs_ppml_coef',
#     sizes=(20, 100),
#     palette={True: 'red', False: 'black'},
#     ax=ax1,
#     legend=False,
#     zorder = 2,
# )
#
# # --- error bars (unchanged) ---
# for _, r in df_plot_1.iterrows():
#     ax1.errorbar(
#         r['diff_elast_minus_ppml'], r['index'],
#         xerr=1.96 * r['se_diff_elast_minus_ppml'],
#         fmt='none',
#         ecolor={True: 'red', False: 'black'}[r['ppml_diff_sig_5']],
#         capsize=5,
#         zorder= 1,
#         alpha = 0.7
#     )
#
# # --- get the actual PathCollection for correct size legend ---
# points = next(c for c in ax1.collections if isinstance(c, mcoll.PathCollection))
# size_handles, size_labels = points.legend_elements(prop="sizes", alpha=1, num=4)
# size_labels = [0, 0.15, 0.30, 0.45, 0.60]
#
# # --- build color + shape legend (clean labels) ---
# sig_color = {True: 'red', False: 'green'}
# color_handles = [
#     Line2D([0], [0], marker='o', linestyle='none', color='none',
#            markerfacecolor=sig_color[True],  markeredgecolor='white', markeredgewidth=0.6,
#            label='Significant (5%)'),
# ]
# shape_handles = [
#     Line2D([0], [0], marker='*', linestyle='none', color='black', label='Sign Changed', markersize=10),
# ]
#
# # --- place legends: main bottom-right, size just above it (also bottom-right) ---
# leg1 = ax1.legend(color_handles + shape_handles,
#                   [h.get_label() for h in color_handles + shape_handles],
#                   title=None, frameon=False, loc='upper left', bbox_to_anchor=(-0.1, 1), ncol=1)
#
# leg2 = ax1.legend(size_handles, size_labels,
#                   title='Absolute value of PPML Coefficient', frameon=False,
#                   loc='upper left', bbox_to_anchor=(-0.1, 0.9), ncol=3)  # adjust 0.18 if overlap
#
# ax1.add_artist(leg1)  # keep both legends visible
# ax1.axvline(x=0, color='black', linestyle='--')
# ax1.get_yaxis().set_visible(False)
# ax1.set_frame_on(False)
# ax1.set_xlabel('')
# fig1.suptitle('Difference between DR Elasticity and PPML Coefficient with 95% CI', weight='bold', fontsize=16)
# ax1.spines['bottom'].set_visible(True)
# ax1.set_xlim(-0.6, 0.6)
# plt.tight_layout()
# plt.savefig(r"C:\Users\Meet Shah\Desktop\Applied Micro Pres\Figures\diff_graph_ppml.pdf", dpi=1000, transparent=True)


# NEW THE GRAPH

df_fig1 = df.query('abs(elasticity) < 10 and abs(ols_coef) < 10 and nobs> 100')
df_fig1.sort_values(by='diff_elast_minus_ols', inplace=True, ascending=True)
df_fig1.reset_index(drop=True, inplace=True)
rows = []
models = ['elast_minus_ols', 'elast_minus_ppml', 'ols_minus_ppml']
p_rows = ['test_elast_vs_ols_p', 'test_elast_vs_ppml_p', 'test_ols_vs_ppml_p']



for index, result in df_fig1.iterrows():
    for i, model in enumerate(models):
        row = {}
        row['model'] = model
        row['diff'] = result[f'diff_{model}']
        row['se_diff'] = result[f'se_diff_{model}']
        row['sig'] = result[p_rows[i]] < 0.05
        row['index'] = index
        row['elasticity'] = result['elasticity']
        row['ols_coef'] = result['ols_coef']
        row['ppml_coef'] = result['ppml_coef']
        rows.append(row)

df_plot = pd.DataFrame(rows)

color_map = {
    ("elast_minus_ols", False): COLOR_NONSIG,
    ("elast_minus_ols", True):  COLOR_SIG,
    ("elast_minus_ppml", False): COLOR_NONSIG,
    ("elast_minus_ppml", True):  COLOR_SIG,
}

symbol_map = {
    ("elast_minus_ols", False): "s",
    ("elast_minus_ols", True):  "^",
    ("elast_minus_ppml", False): "s",
    ("elast_minus_ppml", True):  "^",
}

width = 0.5

offsets = {
    "elast_minus_ols": -width,
    "elast_minus_ppml": 0,
    "ols_minus_ppml": width
}

y_base = np.arange(len(df_plot['index'].unique()))


fig, ax = plt.subplots(figsize=(12, 8), ncols=2, sharex=True, sharey=True)
for i, index in enumerate(df_plot['index'].unique()):
    subset = df_plot[df_plot['index'] == index]
    for j, tup in enumerate(subset.iterrows()):
        if j == 2:
            continue
        row = tup[1]
        ax[j].errorbar(
              x=row['diff'],
              y=y_base[i],
              xerr=1.96 * row['se_diff'],
              yerr=0,
              color=color_map[(row['model'], row['sig'])],
              capsize=3,
              label=f"{row['model']} {'(sig)' if row['sig'] else '(not sig)'}",
              marker=symbol_map[(row['model'], row['sig'])],
                markersize=5,
        )

ax[0].set_title('Elasticity - OLS')
ax[1].set_title('Elasticity - PPML')

for a in ax:
    a.axvline(x=0, color='#888888', linestyle='--', linewidth=0.8)
    a.set_ylabel('')
    a.set_xlabel('Difference')
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    a.spines['left'].set_visible(False)

fig.suptitle(r'Differences with 95\% CI', weight='bold', fontsize=16)
ax[0].set_xlim(-1.5, 1.5)
ax[0].yaxis.set_visible(False)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="^", linestyle="none", color=COLOR_SIG, label="Significant Difference"),
    Line2D([0], [0], marker="s", linestyle="none", color=COLOR_NONSIG, label="Insignificant Difference")
]
ax[0].legend(handles=legend_elements, loc="upper left", frameon=False)


plt.tight_layout()
plt.savefig('./paper/outputs/figure_elasticity_vs_both_ci.pdf', transparent=True, dpi=500)



# OLS vs PPML Difference with 95% CI

# Prepare data for OLS-PPML comparison
df_ols_ppml = df_fig1.copy()
df_ols_ppml.sort_values(by='diff_ols_minus_ppml', inplace=True, ascending=True)
df_ols_ppml.reset_index(drop=True, inplace=True)

y_base_ols_ppml = np.arange(len(df_ols_ppml))

fig, ax = plt.subplots(figsize=(8, 8))


# Plot error bars
for i, (_, row) in enumerate(df_ols_ppml.iterrows()):
    sig = row['ols_ppml_diff_sig_5']
    color = COLOR_SIG if sig else COLOR_NONSIG
    marker = '^' if sig else 's'
    
    ax.errorbar(
        x=row['diff_ols_minus_ppml'],
        y=y_base_ols_ppml[i],
        xerr=1.96 * row['se_diff_ols_minus_ppml'],
        yerr=0,
        color=color,
        capsize=4,
        capthick=1.5,
        linewidth=0.8,
        marker=marker,
        markersize=5,
        zorder=2,
    )

ax.axvline(x=0, color='#888888', linestyle='--', linewidth=0.8)
ax.set_xlabel('Difference (OLS - PPML)')
ax.set_ylabel('')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.yaxis.set_visible(False)
ax.set_xlim(-1.5, 1.5)

fig.suptitle(r'OLS vs PPML Difference with 95\% CI', weight='bold', fontsize=16)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="^", linestyle="none", color=COLOR_SIG, label="Significant Difference"),
    Line2D([0], [0], marker="s", linestyle="none", color=COLOR_NONSIG, label="Insignificant Difference")
]
ax.legend(handles=legend_elements, loc="upper left", frameon=False)

plt.tight_layout()
plt.savefig('./paper/outputs/figure_ols_vs_ppml_ci.pdf', transparent=True, dpi=500)


# 2D Comparison PPML vs OLS

fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(data=df_fig1, y='ppml_coef', x='ols_coef', hue='ols_ppml_diff_sig_5', style='ols_ppml_diff_sig_5', palette=[COLOR_NONSIG, COLOR_SIG], markers=['s', '^'], ax=ax, legend=False)
ax.axline((0, 0), slope=1, color='blue', linestyle='-', label='45° line', alpha=0.5)
ax.axvline(x=0, color='#888888', linestyle='--', linewidth=0.8, alpha=0.8)
ax.axhline(y=0, color='#888888', linestyle='--', linewidth=0.8, alpha=0.8)
ax.set_xlabel('OLS Coefficient')
ax.set_ylabel('PPML Coefficient')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
fig.suptitle('PPML vs OLS Coefficient', weight='bold', fontsize=16)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="^", linestyle="none", color=COLOR_SIG, label="Significant Difference"),
    Line2D([0], [0], marker="s", linestyle="none", color=COLOR_NONSIG, label="Insignificant Difference"),
    Line2D([0], [0], linestyle="-", color="blue", alpha=0.5, label=r"45$^\circ$ Line"),
]
ax.legend(handles=legend_elements, loc="upper left", frameon=False)
plt.tight_layout()
plt.savefig('./paper/outputs/figure_ols_vs_ppml_2d.pdf', transparent=True, dpi=500)


# 2D Comparison Elasticity vs OLS

fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(data=df_fig1, y='elasticity', x='ols_coef', hue='ols_diff_sig_5', style='ols_diff_sig_5', palette=[COLOR_NONSIG, COLOR_SIG], markers=['s', '^'], ax=ax, legend=False)
ax.axline((0, 0), slope=1, color='blue', linestyle='-', label='45° line', alpha=0.5)
ax.axvline(x=0, color='#888888', linestyle='--', linewidth=0.8, alpha=0.8)
ax.axhline(y=0, color='#888888', linestyle='--', linewidth=0.8, alpha=0.8)
ax.set_xlabel('OLS Coefficient')
ax.set_ylabel('Elasticity')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
fig.suptitle('DR Elasticity vs OLS Coefficient', weight='bold', fontsize=16)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="^", linestyle="none", color=COLOR_SIG, label="Significant Difference"),
    Line2D([0], [0], marker="s", linestyle="none", color=COLOR_NONSIG, label="Insignificant Difference"),
    Line2D([0], [0], linestyle="-", color="blue", alpha=0.5, label=r"45$^\circ$ Line"),
]
ax.legend(handles=legend_elements, loc="upper left", frameon=False)
plt.tight_layout()
plt.savefig('./paper/outputs/figure_elasticity_vs_ols_2d.pdf', transparent=True, dpi=500)


# 2D Comparison Elasticity vs PPML

fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(data=df_fig1, y='elasticity', x='ppml_coef', hue='ppml_diff_sig_5', style='ppml_diff_sig_5', palette=[COLOR_NONSIG, COLOR_SIG], markers=['s', '^'], ax=ax, legend=False)
ax.axline((0, 0), slope=1, color='blue', linestyle='-', label='45° line', alpha=0.5)
ax.axvline(x=0, color='#888888', linestyle='--', linewidth=0.8, alpha=0.8)
ax.axhline(y=0, color='#888888', linestyle='--', linewidth=0.8, alpha=0.8)
ax.set_xlabel('PPML Coefficient')
ax.set_ylabel('Elasticity')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
fig.suptitle('DR Elasticity vs PPML Coefficient', weight='bold', fontsize=16)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="^", linestyle="none", color=COLOR_SIG, label="Significant Difference"),
    Line2D([0], [0], marker="s", linestyle="none", color=COLOR_NONSIG, label="Insignificant Difference"),
    Line2D([0], [0], linestyle="-", color="blue", alpha=0.5, label=r"45$^\circ$ Line"),
]
ax.legend(handles=legend_elements, loc="upper left", frameon=False)
plt.tight_layout()
plt.savefig('./paper/outputs/figure_elasticity_vs_ppml_2d.pdf', transparent=True, dpi=500)

