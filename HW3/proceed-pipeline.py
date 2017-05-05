import pipeline1 as pl1
import pipeline2 as pl2

X = np.array(['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio', 'age', 'MonthlyIncome', 'NumberOfTimes90DaysLate'])
y = 'SeriousDlqin2yrs'
models_to_run=['RF','LR','DT','KNN','AB','BAG']
grid_size = 'test'

df = pl1.read_data('data/credit-data.csv', 'csv')
pl1.data_overview(df)
df['SeriousDlqin2yrs'] = df['SeriousDlqin2yrs'].astype(np.bool)
pl1.fill_null(df, ['MonthlyIncome'], 'mean')

clfs, grid = pl2.define_clfs_params(grid_size)
pl2.clf_loop(models_to_run, clfs, grid, df[X], df[y], 0.25)
