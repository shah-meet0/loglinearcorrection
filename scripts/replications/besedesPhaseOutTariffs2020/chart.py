import seaborn as sns
import matplotlib as plt

g = sns.scatterplot(df1, x='PPML',y='value', hue='variable')
g.axline([0,0],[1,1])
for x_val in sorted(df1['PPML'].unique()):
    # Get all points with this x value
    points = df1[df1['PPML'] == x_val].sort_values('value')
    # Draw vertical line connecting these points
    plt.plot([x_val] * len(points), points['value'], 'k-', alpha=0.5)

