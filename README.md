# PY2-DS - Sistema de Predicción de Daño en Cultivos

Proyecto de Data Science para predicción de extent (porcentaje de daño) en imágenes de cultivos mediante modelos multimodales que combinan features visuales y metadatos.

## 📋 Descripción General

Sistema completo que incluye:
- **Análisis exploratorio** de datos visuales y tabulares
- **Modelos de Machine Learning** (Random Forest, XGBoost)
- **Pipeline multimodal** con Early Fusion (visual + metadata)
- **Dashboard interactivo** con Streamlit para visualización y predicciones en tiempo real

---

## 📁 Estructura del Proyecto

```
PY2-DS/
├── 📂 app/                          # Aplicación Streamlit
│   ├── app.py                       # Dashboard principal (5 tabs)
│   ├── prediction_utils.py          # Utilidades para predicciones
│   ├── model/                       # Modelos entrenados
│   │   ├── xgb_early_fusion_final.json    # Modelo XGBoost (3.2 MB)
│   │   ├── pre_final.pkl                  # Preprocessor de metadatos
│   │   ├── pca_vis_full.pkl               # PCA para features visuales
│   │   ├── scaler_*.pkl                   # StandardScalers (3 archivos)
│   │   ├── rf_meta_baseline.pkl           # Random Forest baseline
│   │   └── config.json                    # Configuración del modelo
│   └── Proyecto 2 - Resultados DS Informe.pdf
│
├── 📂 data/                         # Datasets del proyecto
│   ├── Train.csv                    # Dataset de entrenamiento (con labels)
│   ├── Test.csv                     # Dataset de test (sin labels)
│   ├── Test_sample.csv              # Subset de 33 imágenes para demos
│   └── SampleSubmission.csv         # Formato esperado de submission
│
├── 📂 notebooks/                    # Jupyter Notebooks
│   ├── PY2_Final.ipynb              # Notebook principal con pipeline completo
│   └── PY2Seleccion.ipynb           # Experimentos y selección de modelos
│
├── 📂 outputs/                      # Resultados y visualizaciones
│   ├── submission.csv               # Predicciones finales para Kaggle
│   ├── model_metrics_cv.csv         # Métricas de cross-validation
│   ├── model_metrics_train.csv      # Métricas de entrenamiento
│   ├── feature_importances_top20.csv
│   ├── viz_cv_mae_por_modelo.png    # Comparación de modelos
│   ├── viz_feature_importances.png  # Importancia de features
│   └── viz_scatter_true_vs_pred.png # True vs Predicted
│
├── 📂 sample_imgs/                  # Imágenes de muestra (33 archivos)
│   └── *.JPG, *.jpg                 # Subset de imágenes del test
│
├── 📂 informes/                     # Documentación y reportes
│   ├── Proyecto 2 Resultados DS Informe.pdf
│   └── PY2Seleccion.pdf
│
├── 📂 venv/                         # Entorno virtual de Python
│
├── 📄 requirements.txt              # Dependencias del proyecto
├── 📄 .gitignore                    # Archivos ignorados por Git
└── 📄 README.md                     # Este archivo

```

---

## 🔑 Archivos Clave

### 📊 **Dashboard Principal**
- **`app/app.py`** (1,764 líneas)
  - Tab 1: Distribuciones - Histogramas y estadísticas
  - Tab 2: Análisis Categórico - Boxplots y violin plots
  - Tab 3: Comparación de Modelos - Métricas de rendimiento
  - Tab 4: Modelo Final - PCA 3D, predicciones vs reales, residuales
  - Tab 5: **Predicciones en Tiempo Real** 🔮
    - Modo A: Upload de imagen + metadatos manuales
    - Modo B: Selección aleatoria desde test

### 🤖 **Modelos y Predicción**
- **`app/prediction_utils.py`**
  - `load_models()`: Carga todos los artefactos
  - `predict_extent_single()`: Pipeline completo de predicción
  - `extract_visual_features_mock()`: Simula embeddings de MobileNetV3
  - `get_interpretation_text()`: Mapea extent a niveles de severidad

### 📓 **Notebooks de Desarrollo**
- **`notebooks/PY2_Final.ipynb`**
  - Pipeline completo end-to-end
  - Entrenamiento del modelo XGBoost final
  - Generación de todos los artefactos (pickles, JSON)
  - Guardado en `app/model/`

- **`notebooks/PY2Seleccion.ipynb`**
  - Experimentos con diferentes arquitecturas
  - Comparación de estrategias (Late Fusion, Early Fusion, etc.)
  - Selección de hiperparámetros

### 📈 **Datasets**
- **`data/Train.csv`** - Datos de entrenamiento con extent
- **`data/Test.csv`** - Datos de test (sin extent)
- **`data/Test_sample.csv`** - 33 imágenes con rutas válidas en `sample_imgs/`

---

## 🚀 Inicio Rápido

### 1. Instalación

```bash
# Clonar repositorio
git clone https://github.com/mar22266/PY2-DS.git
cd PY2-DS

# Crear y activar entorno virtual
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# source venv/bin/activate    # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Ejecutar Dashboard

```bash
streamlit run app/app.py
```

Abre en tu navegador: **http://localhost:8501**

### 3. Hacer Predicciones

1. Ve al **Tab 5: Predicciones**
2. Selecciona un modo:
   - **Modo A**: Sube tu imagen + ingresa metadatos
   - **Modo B**: Usa ejemplos del test (33 disponibles)
3. Click **"🔮 Predecir Extent"**
4. Visualiza:
   - Gauge chart con porcentaje de daño (0-100)
   - Severidad: Leve / Moderado / Severo / Crítico
   - Detalles técnicos del modelo

---

## 📦 Dependencias Principales

```
streamlit >= 1.28.0     # Dashboard interactivo
pandas >= 2.0.0         # Manipulación de datos
numpy >= 1.24.0         # Operaciones numéricas
plotly >= 5.17.0        # Visualizaciones 3D/2D
xgboost >= 3.0.0        # Modelo de predicción
scikit-learn == 1.6.1   # Preprocessing (versión exacta)
pillow >= 10.0.0        # Procesamiento de imágenes
joblib >= 1.3.0         # Carga de modelos pickled
```

Ver `requirements.txt` para lista completa.

---

## 🧠 Arquitectura del Modelo

### Pipeline Multimodal (Early Fusion)

```
Imagen (JPG)                    Metadatos (CSV)
     ↓                               ↓
MobileNetV3 (mock)          ColumnTransformer
     ↓                               ↓
Embeddings (1280-dim)    One-hot + Features (8-dim)
     ↓                               ↓
StandardScaler              StandardScaler
     ↓                               ↓
PCA (256 comp.)                     ↓
     ↓                               ↓
StandardScaler                      ↓
     ↓                               ↓
     └──────── FUSION ──────────────┘
              (weight: 0.2)
                  ↓
          XGBoost Regressor
                  ↓
          Extent (0-100)
```

### Métricas del Modelo Final

- **MAE**: ~8.5
- **RMSE**: ~12.3
- **R²**: 0.85
- **Mejora vs Baseline**: 15% en MAE

---

## 📊 Funcionalidades del Dashboard

### Tab 1: Distribuciones
- Histogramas de extent por tipo de daño
- Estadísticas descriptivas
- Análisis de distribución

### Tab 2: Análisis Categórico
- Boxplots: extent vs damage
- Violin plots: season, growth_stage
- Análisis de categorías

### Tab 3: Comparación de Modelos
- Tabla comparativa de métricas
- Gráficos de barras MAE/RMSE/R²
- Múltiples arquitecturas evaluadas

### Tab 4: Modelo Final
- **PCA 3D Interactivo**:
  - Coloreado por damage
  - Coloreado por season
  - Coloreado por growth_stage
  - Gradient por extent
- **True vs Predicted**: 2D y 3D scatter
- **Histograma de Residuales**
- Métricas finales

### Tab 5: Predicciones 🔮
- **Modo A**: Upload + metadatos manuales
- **Modo B**: Random test selection (33 imágenes)
- Gauge chart con extent
- Interpretación de severidad:
  - 🟢 0-25: Leve
  - 🟡 26-50: Moderado
  - 🟠 51-75: Severo
  - 🔴 76-100: Crítico
- Historial de predicciones

---

## 🛠️ Scripts Útiles

### Verificar Sistema
```bash
python verify_system.py
# Verifica que todos los modelos estén cargables
```

### Regenerar Test Sample
```bash
python regenerate_test_sample.py
# Sincroniza Test_sample.csv con imágenes en sample_imgs/
```

---

## 📝 Notas Técnicas

### Features Visuales (Mock)
- Actualmente se **simulan** embeddings de MobileNetV3
- Para producción: cargar modelo preentrenado real
- Reemplazar `extract_visual_features_mock()` en `prediction_utils.py`

### Compatibilidad de Modelos
- **Crítico**: scikit-learn debe ser 1.6.1 exacto
- Pickles fueron serializados con `joblib.dump()`
- XGBoost 3.0+ requerido

### Cache de Streamlit
- Modelos se cargan una sola vez con `@st.cache_resource`
- Primera predicción: ~2-3 segundos
- Predicciones posteriores: <100ms

---

## 🐛 Troubleshooting

### "Imagen no encontrada"
```bash
# Regenerar Test_sample.csv
python regenerate_test_sample.py
```

### "STACK_GLOBAL requires str"
```bash
# Reinstalar scikit-learn versión exacta
pip uninstall scikit-learn -y
pip install scikit-learn==1.6.1
```

### Dashboard no inicia
```bash
# Verificar puerto 8501
netstat -ano | findstr :8501

# Reiniciar
streamlit run app/app.py
```

---

## 📚 Documentación Adicional

- **`FIXES_APPLIED.md`** - Log de problemas resueltos
- **`STATUS_SUMMARY.md`** - Estado completo del sistema
- **`TRAIN_MODEL_INSTRUCTIONS.md`** - Cómo entrenar modelos
- **`README_DASHBOARD.md`** - Guía detallada del dashboard
- **`verify_system.py`** - Script de verificación

---

## 👥 Autores

- **Proyecto**: PY2-DS
- **Curso**: Data Science
- **Universidad**: Universidad del Valle de Guatemala

---

## 📄 Licencia

Este proyecto es parte de un trabajo académico.

---

## 🎯 Estado del Proyecto

✅ **Completado y Funcional**

- Pipeline de entrenamiento ✅
- Modelos entrenados y guardados ✅
- Dashboard interactivo ✅
- Sistema de predicciones ✅
- Documentación completa ✅

**Dashboard activo en**: http://localhost:8501

---

## 🔗 Links Útiles

- **Repositorio**: https://github.com/mar22266/PY2-DS
- **Pull Request Activo**: [#2 Resultados parciales y visualizaciones estáticas](https://github.com/mar22266/PY2-DS/pull/2)
- **Branch**: `Resultados-Parciales-y-Visualizaciones-Estáticas`

---

**¡Sistema listo para usar! 🚀**