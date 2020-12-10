# task3
Decision-trees
Importing libraries in Python
import sklearn.datasets as datasets import pandas as pd
Loading the iris dataset
iris=datasets.load_iris()
Forming the iris dataframe
df=pd.DataFrame(iris.data, columns=iris.feature_names) print(df.head(5))
y=iris.target print(y)
Defining the decision tree algorithm
from sklearn.tree import DecisionTreeClassifier dtree=DecisionTreeClassifier() dtree.fit(df,y)
print('Decision Tree Classifer Created')
Install required libraries
!pip install pydotplus !apt-get install graphviz -y
Import necessary libraries for graph viz
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz import pydotplus
Visualize the graph
dot_data = StringIO() export_graphviz(dtree, out_file=dot_data, feature_names=iris.feature_names,
filled=True, rounded=True, special_characters=True) graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
