import pandas as pd


#MODEL='M_201028_FU_23'
MODEL='M_201028_FU_M10'
TOPIC='10'

file = 'M10_T'+TOPIC+'_coded.csv'
df = pd.read_csv('./coded/'+file, engine='python')

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

sns.distplot(df['P'], hist=False, rug=False, kde=True,
             bins=100, color = 'blue', label = 'All',
             hist_kws={'edgecolor':'black'},
             kde_kws = {'shade': True, 'linewidth': 2})

df_M = df[(df['CODING'] == "M")]
sns.distplot(df_M['P'], hist=False, rug=False, kde=True,
             bins=100, color = 'green', label = 'Marketing',
             hist_kws={'edgecolor':'black'})

df_E = df[(df['CODING'] == "E")]
sns.distplot(df_E['P'], hist=False, rug=False, kde=True,
             bins=100, color = 'yellow', label = 'Entrepreneur',
             hist_kws={'edgecolor':'black'})

df_Y = df[(df['CODING'] == "Y")]
sns.distplot(df_Y['P'], hist=False, rug=False, kde=True,
             bins=100, color = 'red', label = 'Marketing & Entrepreneur',
             hist_kws={'edgecolor':'black'})

#Add legend
plt.legend(prop={'size': 10}, title = 'Set')

# Add labels
plt.title('Match probabilities Coded, model= ' + MODEL + ' , T= ' + TOPIC)
plt.xlabel('Probability')
plt.ylabel('Articles')

plt.tight_layout()
plt.show()
