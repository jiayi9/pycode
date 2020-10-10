
import numpy as np
import matplotlib.pyplot as plt
x = np.random.randint(low=0, high=100, size=100)

#frequency, bins = np.histogram(x, bins=10, range=[0, 100])

plt.hist(x)

plt.hist(x, bins = 50)



import pandas as pd
#df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/diamonds.csv')
df.head()

# histgram in stupid way
x1 = df.loc[df.cut=='Ideal', 'depth']
x2 = df.loc[df.cut=='Fair', 'depth']
x3 = df.loc[df.cut=='Good', 'depth']

kwargs = dict(alpha=0.5, bins=100)

plt.hist(x1, **kwargs, color='g', label='Ideal')
plt.hist(x2, **kwargs, color='b', label='Fair')
plt.hist(x3, **kwargs, color='r', label='Good')
plt.gca().set(title='Frequency Histogram of Diamond Depths', ylabel='Frequency')
plt.xlim(50,75)
plt.legend()


# normalized histogram in stupid way

kwargs = dict(alpha=0.5, bins=100, density=True, stacked=True)

# Plot
plt.hist(x1, **kwargs, color='g', label='Ideal')
plt.hist(x2, **kwargs, color='b', label='Fair')
plt.hist(x3, **kwargs, color='r', label='Good')
plt.gca().set(title='Probability Histogram of Diamond Depths', ylabel='Probability')
plt.xlim(50,75)
plt.legend()




########################## bery stupid way

fig, axes = plt.subplots(1, 5, figsize=(10,2.5), dpi=100, sharex=True, sharey=True)

colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:pink', 'tab:olive']

for i, (ax, cut) in enumerate(zip(axes.flatten(), df.cut.unique())):
    x = df.loc[df.cut==cut, 'depth']
    ax.hist(x, alpha=0.5, bins=100, density=True, stacked=True, label=str(cut), color=colors[i])
    ax.set_title(cut)

plt.suptitle('Probability Histogram of Diamond Depths', y=1.05, size=16)
ax.set_xlim(50, 70); ax.set_ylim(0, 1);
plt.tight_layout();


###########################
kwargs = dict(alpha=0.5, bins=100, density=True, stacked=True)

df['x'].hist(by=df['cut'], **kwargs)














##############  matplotlib with seaborn  #################

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="ticks", color_codes=True)

tips = sns.load_dataset("tips")

tips.head(10)

g = sns.FacetGrid(tips, col="time", row="smoker")

# run the following two rows together
g = sns.FacetGrid(tips, col="time",  row="smoker")
g = g.map(plt.hist, "total_bill")

# is equal to
sns.FacetGrid(tips, col="time",  row="smoker",hue = 'day').map(plt.hist, "total_bill", alpha = 0.8)






import numpy as np
bins = np.arange(0, 65, 5)

g = sns.FacetGrid(tips, col="time",  row="smoker")
g = g.map(plt.hist, "total_bill", bins=bins, color="r")

g = sns.FacetGrid(tips, col="time",  row="smoker")
g = g.map(plt.hist, "total_bill", bins=30, color="r")

g = sns.FacetGrid(tips, col="time",  row="smoker")
g = g.map(plt.hist, "total_bill", bins=np.array([5,30]), color="r")


# geom_scatter + facet_wrap
g = sns.FacetGrid(tips, col="time",  row="smoker")
g = g.map(plt.scatter, "total_bill", "tip", edgecolor="w")

# hep,_scatter + color + facet_wrap
g = sns.FacetGrid(tips, col="time",  hue="smoker")
g = (g.map(plt.scatter, "total_bill", "tip", edgecolor="w").add_legend())

sns.FacetGrid(tips, col="time",  hue="smoker").map(plt.scatter, "total_bill", "tip", edgecolor="w").add_legend()

sns.FacetGrid(tips, col="time", row = "smoker", hue="size").map(plt.scatter, "total_bill", "tip", edgecolor="w").add_legend()

# error
g = sns.FacetGrid(tips, col="day", height=4, aspect=.5)
g = g.map(plt.hist, "total_bill", bins=bins)

# no error
g = sns.FacetGrid(tips, col="day", size=4, aspect=.5)
g = g.map(plt.hist, "total_bill", bins=bins)


# factor order
g = sns.FacetGrid(tips, col="smoker", col_order=["Yes", "No"])
g = g.map(plt.hist, "total_bill", bins=bins, color="m")


# theme
kws = dict(s=50, linewidth=.5, edgecolor="w")
g = sns.FacetGrid(tips, col="sex", hue="time", palette="Set1",hue_order=["Dinner", "Lunch"])
g = (g.map(plt.scatter, "total_bill", "tip", **kws).add_legend())



g = sns.FacetGrid(tips, col="sex", hue="time", palette=pal,
                   hue_order=["Dinner", "Lunch"],
                   hue_kws=dict(marker=["^", "v"]))
g = (g.map(plt.scatter, "total_bill", "tip", **kws)
      .add_legend())



att = sns.load_dataset("attention")
g = sns.FacetGrid(att, col="subject", col_wrap=5, size=1.5)
g = g.map(plt.plot, "solutions", "score", marker=".")



#from scipy import stats
#def qqplot(x, y, **kwargs):
#     _, xr = stats.probplot(x, fit=False)
#     _, yr = stats.probplot(y, fit=False)
#     sns.scatterplot(xr, yr, **kwargs)
#g = sns.FacetGrid(tips, col="smoker", hue="sex")
#g = (g.map(qqplot, "total_bill", "tip", **kws)
#       .add_legend())




import pandas as pd
df = pd.DataFrame(
     data=np.random.randn(90, 4),
     columns=pd.Series(list("ABCD"), name="walk"),
     index=pd.date_range("2015-01-01", "2015-03-31",
                         name="date"))
df = df.cumsum(axis=0).stack().reset_index(name="val")
def dateplot(x, y, **kwargs):
     ax = plt.gca()
     data = kwargs.pop("data")
     data.plot(x=x, y=y, ax=ax, grid=False, **kwargs)
g = sns.FacetGrid(df, col="walk", col_wrap=2, size=3.5)
g = g.map_dataframe(dateplot, "date", "val")



g = sns.FacetGrid(tips, col="smoker", row="sex")
g = (g.map(plt.scatter, "total_bill", "tip", color="g", **kws)
       .set_axis_labels("Total bill (US Dollars)", "Tip"))




g = sns.FacetGrid(tips, col="smoker", row="sex")
g = (g.map(plt.scatter, "total_bill", "tip", color="r", **kws)
       .set(xlim=(0, 60), ylim=(0, 12),
            xticks=[10, 30, 50], yticks=[2, 6, 10]))


g = sns.FacetGrid(tips, col="size", col_wrap=3)
g = (g.map(plt.hist, "tip", bins=np.arange(0, 13), color="c")
       .set_titles("{col_name} diners"))



g = sns.FacetGrid(tips, col="smoker", row="sex",
                   margin_titles=True)
g = (g.map(plt.scatter, "total_bill", "tip", color="m", **kws)
       .set(xlim=(0, 60), ylim=(0, 12),
            xticks=[10, 30, 50], yticks=[2, 6, 10])
       .fig.subplots_adjust(wspace=.05, hspace=.05))
       
       
       
# 
       
import numpy as np; np.random.seed(10)
import seaborn as sns; sns.set(color_codes=True)
mean, cov = [0, 2], [(1, .5), (.5, 1)]
x, y = np.random.multivariate_normal(mean, cov, size=50).T

ax = sns.kdeplot(x)

ax = sns.kdeplot(x, shade=True, color="r")

ax = sns.kdeplot(x, y)

ax = sns.kdeplot(x, y, shade=True)

ax = sns.kdeplot(x, y, n_levels=30, cmap="Purples_d")

ax = sns.kdeplot(x, bw=.15)

ax = sns.kdeplot(x, cut=0)


plt.figure() # Push new figure on stack
ax = sns.kdeplot(x, y, cbar=True)
#ax.savefig("C:/daten/seaborn.jpg")
plt.tight_layout()
plt.savefig("C:/daten/seaborn_tight_dpi199.jpg", dpi = 199)


iris = sns.load_dataset("iris")
setosa = iris.loc[iris.species == "setosa"]
virginica = iris.loc[iris.species == "virginica"]
ax = sns.kdeplot(setosa.sepal_width, setosa.sepal_length,
                  cmap="Reds", shade=True, shade_lowest=False)
ax = sns.kdeplot(virginica.sepal_width, virginica.sepal_length,
                  cmap="Blues", shade=True, shade_lowest=False)


sns.pairplot(iris, hue="species");







# https://seaborn.pydata.org/tutorial/categorical.html

tips = sns.load_dataset("tips")
sns.catplot(x="day", y="total_bill", data=tips)

sns.catplot(x="day", y="total_bill", jitter=False, data=tips);



###########################################################################

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

array = [[13,1,1,0,2,0],
         [3,9,6,0,1,0],
         [0,0,16,2,0,0],
         [0,0,0,13,0,0],
         [0,0,0,0,15,0],
         [0,0,1,0,0,15]]

df_cm = pd.DataFrame(array, range(6), range(6))
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

plt.show()








