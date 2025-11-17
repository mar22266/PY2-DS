# Script para generar Test_sample.csv solo con imágenes que existan en sample_imgs

import pandas as pd
import os

# Cargar Test.csv
df_test = pd.read_csv('Test.csv')

# Obtener lista de archivos en sample_imgs
sample_files = set(os.listdir('sample_imgs'))

# Filtrar solo las filas donde el archivo existe en sample_imgs
df_test_filtered = df_test[df_test['filename'].isin(sample_files)]

# Si hay menos de 100, tomar todas; si hay más, tomar 100 aleatorias
if len(df_test_filtered) > 100:
    df_sample = df_test_filtered.sample(n=100, random_state=42)
else:
    df_sample = df_test_filtered

# Guardar
df_sample.to_csv('Test_sample.csv', index=False)

print(f"✅ Test_sample.csv generado con {len(df_sample)} filas")
print(f"Total de imágenes disponibles en sample_imgs: {len(sample_files)}")
print(f"Imágenes del Test.csv que existen en sample_imgs: {len(df_test_filtered)}")
print(f"\nPrimeras 5 filas:")
print(df_sample.head())
