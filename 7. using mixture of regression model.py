import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
 dat is named 'Like.n', an
mcdonalds['Like.n'] = 6 - pd.to_numeric(mcdonalds['Like'])

independent_variables = mcdonalds.columns[1:12]
formula = 'Like.n ~ ' + ' + '.join(independent_variables) 

model = sm.OLS.from_formula(formula, data=mcdonalds)
result = model.fit()

print(result.summary())

plt.figure(figsize=(10, 6))
coefs = result.params.drop('Intercept')
pvalues = result.pvalues.drop('Intercept')
significance = pvalues < 0.05  

colors = ['grey' if sig else 'lightgrey' for sig in significance]
error_bars = [result.conf_int()[0], result.conf_int()[1]]

coefs.plot(kind='bar', color=colors, yerr=error_bars, capsize=5)
plt.xlabel('Independent Variables')
plt.ylabel('Coefficient Value')
plt.title('Regression Coefficients with Significance')
plt.show()