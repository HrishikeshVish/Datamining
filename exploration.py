import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
np.random.seed(0)


raw_data = pd.read_csv('digits-raw.csv')
columns = ['image_id', 'class']
for i in range(784):
    columns.append('pixel_feature'+str(i))
raw_data.columns = columns

zero = np.asarray(raw_data.loc[raw_data['class']==0].sample(n=1, random_state=0).drop('image_id', axis=1).drop('class', axis=1)).reshape(28,28)
one = np.asarray(raw_data.loc[raw_data['class']==1].sample(n=1, random_state=0).drop('image_id', axis=1).drop('class', axis=1)).reshape(28,28)
two = np.asarray(raw_data.loc[raw_data['class']==2].sample(n=1, random_state=0).drop('image_id', axis=1).drop('class', axis=1)).reshape(28,28)
three = np.asarray(raw_data.loc[raw_data['class']==3].sample(n=1, random_state=0).drop('image_id', axis=1).drop('class', axis=1)).reshape(28,28)
four = np.asarray(raw_data.loc[raw_data['class']==4].sample(n=1, random_state=0).drop('image_id', axis=1).drop('class', axis=1)).reshape(28,28)
five = np.asarray(raw_data.loc[raw_data['class']==5].sample(n=1, random_state=0).drop('image_id', axis=1).drop('class', axis=1)).reshape(28,28)
six = np.asarray(raw_data.loc[raw_data['class']==6].sample(n=1, random_state=0).drop('image_id', axis=1).drop('class', axis=1)).reshape(28,28)
seven = np.asarray(raw_data.loc[raw_data['class']==7].sample(n=1, random_state=0).drop('image_id', axis=1).drop('class', axis=1)).reshape(28,28)
eight = np.asarray(raw_data.loc[raw_data['class']==8].sample(n=1, random_state=0).drop('image_id', axis=1).drop('class', axis=1)).reshape(28,28)
nine = np.asarray(raw_data.loc[raw_data['class']==9].sample(n=1, random_state=0).drop('image_id', axis=1).drop('class', axis=1)).reshape(28,28)

fig = plt.figure()
grid = ImageGrid(fig, 111, nrows_ncols=(3,4), axes_pad=0.1,)
for ax, im in zip(grid, [zero, one, two, three, four, five, six, seven, eight, nine]):
    ax.imshow(im)
plt.show()

embedding = pd.read_csv('digits-embedding.csv')
columns = ['image_id', 'class', 'embed1', 'embed2']
embedding.columns = columns
rand_samples = np.random.randint(0, len(embedding), size=1000)
samples = embedding.iloc[rand_samples]

x_coordinates = samples['embed1']
y_coordinates = samples['embed2']
classes = samples['class']
color_dict = {0:'yellow', 1:'blue', 2:'green', 3:'red', 4:'black', 5:'brown', 6:'violet', 7:'pink', 8:'gray', 9:'orange'}
colors = []
for i in classes:
    colors.append(color_dict[i])

plt.scatter(x_coordinates, y_coordinates, color=colors)
plt.show()
