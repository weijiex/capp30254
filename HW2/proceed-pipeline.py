

import pipeline as pl




y = 'SeriousDlqin2yrs'
cols = list(df.columns.values)
X = [x for x in cols if x != y]
features = X


# # Step 1 & 2: Read and Explore Data



df = pl.read_data('credit-data.csv', 'csv')
pl.data_overview(df)




for col_name in X:
    pl.make_graph(df, col_name)


# # Step 3: Pre-Process Data



pl.fill_null(df, 'MonthlyIncome', 'mean')
pl.fill_null(df, 'NumberOfDependents', 'median')
df.isnull().sum()


# # Step 4: Generate Features/Predictors



df = pl.discretize_cont_var(df, 'NumberOfDependents', num_bins=3, cut_type='uniform', labels=['low','med','high'])
df = pl.binarize_categ_var(df, 'NumberOfDependents_discretize')




df = pl.discretize_cont_var(df, 'MonthlyIncome', num_bins=5, cut_type='quantile',                            labels=['low_level','med minus','med_level','med plus','high_level'])
df = pl.binarize_categ_var(df, 'MonthlyIncome_discretize')




df.head()


# # Step 5: Build Classifier



from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier




X_train, X_test, y_train, y_test = pl.split_data(df, X, y, 0.2)




classifiers = [LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier(),                RandomForestClassifier(), GradientBoostingClassifier()]




for method in classifiers:
    pl.test_model(X_train, y_train, features, method)
    pl.predict_model(X_train, y_train, X_test, y, features, method)


# # Step 6: Evaluate Classifier



for method in classifiers:
    pl.eval_model(X_train, y_train, X_test, y_test, features, method)






