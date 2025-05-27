import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('dados.csv')
df.columns = ['salario_anual', 'total_dividas', 'historico_pagamento', 'idade', 'credito_solicitado', 'elegibilidade']

def limpar_historico(valor):
    try:
        return float(valor)
    except:
        return np.nan

df['historico_pagamento'] = df['historico_pagamento'].apply(limpar_historico)
df.dropna(inplace=True)

def remover_outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lim_inf = q1 - 1.5 * iqr
    lim_sup = q3 + 1.5 * iqr
    return df[(df[col] >= lim_inf) & (df[col] <= lim_sup)]

for c in ['salario_anual', 'total_dividas', 'credito_solicitado']:
    df = remover_outliers(df, c)

X = df[['salario_anual', 'total_dividas', 'historico_pagamento', 'idade', 'credito_solicitado']]
y = df['elegibilidade']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=30)

modelo = KNeighborsClassifier(n_neighbors=9)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f'Precisão: {acc * 100:.2f}%')

joblib.dump(modelo, 'modelo.joblib')
joblib.dump(scaler, 'scaler.joblib')

entrada = [[50000, 20000, 8.5, 35, 15000]]
entrada_df = pd.DataFrame(entrada, columns=X.columns)
entrada_scaled = scaler.transform(entrada_df)
res = modelo.predict(entrada_scaled)

print(f'previsão para a entrada {entrada} -> {res[0]}')