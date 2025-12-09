import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Descargar matriz insumo-producto del INEGI
# https://www.inegi.org.mx/programas/mip/2018/#datos_abiertos

dir_arch = 'mip_csv/conjunto_de_datos/conjunto_de_datos_mip_cdi_ixi_12018.csv'
matriz_io = pd.read_csv(dir_arch, index_col=0)

# Limpiar datos
matriz_io = matriz_io.iloc[:, :matriz_io.shape[0]]
matriz_io = matriz_io.apply(pd.to_numeric, errors='coerce').fillna(0)

# Calcular matriz de coeficientes técnicos (A)
produccion_total = matriz_io.sum(axis=0)
A = matriz_io.div(produccion_total, axis=1).fillna(0)

# Calcular matriz de Leontief (I - A)^-1
I = np.eye(A.shape[0])
L = np.linalg.inv(I - A.values)

# Seleccionar sector para choque de demanda
sectores = matriz_io.index
print("\nSectores disponibles:")
for i, sector in enumerate(sectores):
    print(f"{i}: {sector}")

#sector_idx = int(input("\nIngrese el índice del sector: "))
#aumento_demanda = float(input("Ingrese el aumento en demanda (millones de pesos): "))
sector_idx = 3  # Ejemplo: Construcción
aumento_demanda = 10000.0  # Ejemplo: 10,000 millones de pesos 


print(f"\nSector seleccionado: {sectores[sector_idx]}")

delta_demanda = np.zeros(len(sectores))
delta_demanda[sector_idx] = aumento_demanda

# Calcular impacto total en producción
impacto_produccion = L @ delta_demanda

# Mostrar resultados
resultados = pd.DataFrame({
    'Sector': sectores,
    'Impacto en Producción': impacto_produccion
}).sort_values('Impacto en Producción', ascending=False)

print(f"\n{'='*80}")
print(f"IMPACTO DE AUMENTO EN DEMANDA: ${aumento_demanda:,.0f} millones")
print(f"Sector: {sectores[sector_idx]}")
print(f"{'='*80}")
print(f"\nTop 10 sectores más impactados:")
print(resultados.head(10).to_string(index=False))
print(f"\nImpacto total en la economía: ${impacto_produccion.sum():,.2f} millones")
print(f"Multiplicador: {impacto_produccion.sum()/aumento_demanda:.2f}")

# Guardar resultados
resultados.to_csv('impacto_demanda.csv', index=False)
print("\nResultados guardados en 'impacto_demanda.csv'")

# Gráficas
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gráfica 1: Top 15 sectores más impactados
top15 = resultados.head(15)
axes[0].barh(range(len(top15)), top15['Impacto en Producción'], color='steelblue')
axes[0].set_yticks(range(len(top15)))
axes[0].set_yticklabels([s[:40] for s in top15['Sector']], fontsize=9)
axes[0].invert_yaxis()
axes[0].set_xlabel('Impacto (millones de pesos)', fontsize=11)
axes[0].set_title('Top 15 Sectores Más Impactados', fontsize=13, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

# Gráfica 2: Distribución del impacto
impactos_positivos = resultados[resultados['Impacto en Producción'] > 0.01]
axes[1].hist(impactos_positivos['Impacto en Producción'], bins=30, color='coral', edgecolor='black')
axes[1].set_xlabel('Impacto en Producción (millones)', fontsize=11)
axes[1].set_ylabel('Número de Sectores', fontsize=11)
axes[1].set_title('Distribución del Impacto Económico', fontsize=13, fontweight='bold')
axes[1].axvline(impactos_positivos['Impacto en Producción'].mean(), 
                color='red', linestyle='--', linewidth=2, label='Media')
axes[1].legend()

plt.tight_layout()
plt.savefig('impacto_demanda.png', dpi=300, bbox_inches='tight')
print("Gráficas guardadas en 'impacto_demanda.png'")
plt.show()
