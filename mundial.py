import pandas as pd
import numpy as np

# Cargar matriz insumo-producto
matriz_io = pd.read_csv('mip/conjunto/conjunto_de_datos_mip_cdi_ixi_22018.csv', index_col=0)

# Limpiar datos
matriz_io = matriz_io.iloc[:, :matriz_io.shape[0]]
matriz_io = matriz_io.apply(pd.to_numeric, errors='coerce').fillna(0)

# Calcular matriz de coeficientes técnicos (A)
produccion_total = matriz_io.sum(axis=0)
A = matriz_io.div(produccion_total, axis=1).fillna(0)

# Calcular transformada inversa de Leontief (I - A)^-1
I = np.eye(A.shape[0])
L = np.linalg.inv(I - A.values)

# Guardar resultado
leontief_df = pd.DataFrame(L, index=matriz_io.index, columns=matriz_io.columns)
leontief_df.to_csv('transformada_leontief.csv')

print(f"Transformada inversa de Leontief calculada: {L.shape}")
print("Guardada en 'transformada_leontief.csv'")