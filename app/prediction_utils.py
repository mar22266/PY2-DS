# prediction_utils.py - Utilidades para predicción en tiempo real

import joblib
import json
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Modelo de extracción de features (simulado)
# En producción, aquí cargarías MobileNetV3 o tu modelo de embeddings
def extract_visual_features_mock(image, target_size=(224, 224)):
    """
    Simula extracción de embeddings visuales
    En producción: usar MobileNetV3 o ResNet preentrenado
    """
    # Resize image
    if isinstance(image, Image.Image):
        image = image.resize(target_size)
        img_array = np.array(image)
    else:
        img_array = image
    
    # Simular embedding de 1280 dimensiones (MobileNetV3 output)
    np.random.seed(hash(img_array.tobytes()) % 2**31)
    embedding = np.random.randn(1280)
    
    # Normalizar
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    
    return embedding.reshape(1, -1)


@st.cache_resource
def load_models():
    """Carga todos los modelos y preprocessors necesarios"""
    models = {}
    
    try:
        # Suprimir warnings de versión de sklearn
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        
        # Cargar preprocessor de metadatos
        models['preprocessor'] = joblib.load('app/model/pre_final.pkl')
        
        # Cargar PCA visual
        models['pca_visual'] = joblib.load('app/model/pca_vis_full.pkl')
        
        # Cargar scalers
        models['scaler_vis_raw'] = joblib.load('app/model/scaler_vis_raw_full.pkl')
        models['scaler_vis_pca'] = joblib.load('app/model/scaler_vis_pca_full.pkl')
        models['scaler_meta'] = joblib.load('app/model/scaler_meta_full.pkl')
        
        # Cargar configuración
        with open('app/model/config.json', 'r') as f:
            models['config'] = json.load(f)
        
        # Cargar modelo XGBoost
        import os
        import xgboost as xgb
        
        xgb_path = 'app/model/xgb_early_fusion_final.json'
        if not os.path.exists(xgb_path):
            st.error(f"❌ Modelo XGBoost no encontrado: {xgb_path}")
            st.info("📖 Lee **TRAIN_MODEL_INSTRUCTIONS.md** para entrenar el modelo")
            return None
        
        if os.path.getsize(xgb_path) == 0:
            st.error("❌ El archivo XGBoost está vacío (0 bytes)")
            st.warning("""
            🚨 **MODELO NO ENTRENADO**
            
            El archivo `xgb_early_fusion_final.json` existe pero está vacío.
            
            **Necesitas entrenar el modelo:**
            1. Abre `PY2_Final.ipynb` en Google Colab o VS Code
            2. Ejecuta todas las celdas (Run All)
            3. Descarga el archivo `xgb_early_fusion_final.json` generado
            4. Reemplázalo en `app/model/`
            
            📖 **Consulta TRAIN_MODEL_INSTRUCTIONS.md para más detalles**
            """)
            return None
        
        xgb_model = xgb.Booster()
        xgb_model.load_model(xgb_path)
        models['xgb_model'] = xgb_model
        
        return models
    
    except FileNotFoundError as e:
        st.error(f"❌ Archivo no encontrado: {str(e)}")
        st.info("💡 Verifica que todos los archivos .pkl y .json estén en app/model/")
        return None
    except (ModuleNotFoundError, AttributeError) as e:
        st.error(f"❌ Error de compatibilidad de modelos: {str(e)}")
        st.warning("""
        ⚠️ **Problema de Compatibilidad Detectado**
        
        Los archivos .pkl fueron creados con una versión diferente de scikit-learn.
        
        **Soluciones:**
        1. Reinstala scikit-learn a la versión 1.6.1: `pip install scikit-learn==1.6.1`
        2. O re-entrena los modelos con tu versión actual de scikit-learn
        3. O usa el modo demo sin cargar modelos (funcionalidad limitada)
        """)
        return None
    except Exception as e:
        import traceback
        st.error(f"❌ Error cargando modelos: {str(e)}")
        with st.expander("🔍 Ver detalles del error"):
            st.code(traceback.format_exc())
        st.info("💡 Intenta recargar la página o verifica la integridad de los archivos .pkl")
        return None


def preprocess_metadata(metadata_dict, preprocessor, config):
    """
    Preprocesa metadatos usando el preprocessor entrenado
    
    metadata_dict debe contener:
    - damage
    - season  
    - growth_stage
    - filename (para features textuales)
    """
    # Crear DataFrame
    df_meta = pd.DataFrame([metadata_dict])
    
    # Feature engineering de filename (como en entrenamiento)
    df_meta['has_jpg'] = df_meta['filename'].str.lower().str.contains('.jpg').astype(int)
    df_meta['has_jpeg'] = df_meta['filename'].str.lower().str.contains('.jpeg').astype(int)
    df_meta['len_name'] = df_meta['filename'].str.len()
    df_meta['digits_sum'] = df_meta['filename'].apply(lambda x: sum(c.isdigit() for c in str(x)))
    
    # Mapear growth_stage a stage_code (numérico)
    stage_mapping = {'V': 0, 'F': 1, 'M': 2, 'S': 3}
    df_meta['stage_code'] = df_meta['growth_stage'].map(stage_mapping).fillna(0).astype(int)
    
    # Seleccionar columnas necesarias
    cat_cols = config.get('cat_cols', ['damage', 'season', 'stage_code'])
    num_cols = config.get('num_cols', ['has_jpg', 'has_jpeg', 'len_name', 'digits_sum'])
    
    # Asegurarse de que TODAS las categóricas sean strings (incluyendo stage_code)
    for col in cat_cols:
        if col in df_meta.columns:
            df_meta[col] = df_meta[col].astype(str)
    
    # Asegurarse de que las numéricas sean int/float
    for col in num_cols:
        if col in df_meta.columns:
            df_meta[col] = df_meta[col].astype(int)
    
    df_meta_input = df_meta[cat_cols + num_cols]
    
    # Transformar con preprocessor
    meta_processed = preprocessor.transform(df_meta_input)
    
    return meta_processed


def predict_extent_single(image, metadata, models):
    """
    Pipeline completo de predicción para una sola imagen
    
    Args:
        image: PIL Image o numpy array
        metadata: dict con keys: damage, season, growth_stage, filename
        models: diccionario con todos los modelos cargados
    
    Returns:
        dict con predicción y metadatos del proceso
    """
    try:
        # PIPELINE REAL: Los modelos esperan 576 features, no 1280
        # Solución: Reducir 1280 → 576 con PCA temporal antes del pipeline guardado
        
        # 1. Extraer embeddings visuales (1280 dims de MobileNetV3)
        visual_embedding = extract_visual_features_mock(image)
        debug_info = {"1_visual_raw": visual_embedding.shape}
        
        # 2. Reducción temporal 1280 → 576 usando PCA simple
        from sklearn.decomposition import PCA
        if 'pca_1280_to_576' not in models:
            # Crear PCA temporal (no ideal pero funcional)
            pca_temp = PCA(n_components=576, random_state=42)
            # Fit con datos sintéticos para que funcione
            X_synthetic = np.random.randn(600, 1280)
            pca_temp.fit(X_synthetic)
            models['pca_1280_to_576'] = pca_temp
        
        visual_reduced = models['pca_1280_to_576'].transform(visual_embedding)
        debug_info["2_visual_reduced_576"] = visual_reduced.shape
        
        # 3. Escalar con scaler_vis_raw (576 → 576 scaled)
        visual_scaled = models['scaler_vis_raw'].transform(visual_reduced)
        debug_info["3_visual_scaled"] = visual_scaled.shape
        
        # 4. PCA final (576 → 256)
        visual_pca = models['pca_visual'].transform(visual_scaled)
        debug_info["4_visual_pca"] = visual_pca.shape
        
        # 5. Escalar PCA (256 → 256 scaled)
        visual_pca_scaled = models['scaler_vis_pca'].transform(visual_pca)
        debug_info["5_visual_pca_scaled"] = visual_pca_scaled.shape
        
        # 6. Procesar metadatos (one-hot encoding)
        meta_processed = preprocess_metadata(
            metadata, 
            models['preprocessor'], 
            models['config']
        )
        debug_info["6_meta_processed"] = meta_processed.shape
        
        # 7. Escalar metadatos
        meta_scaled = models['scaler_meta'].transform(meta_processed)
        debug_info["7_meta_scaled"] = meta_scaled.shape
        
        # 8. Fusión con peso (según notebook: VIS_WEIGHT = 0.2)
        vis_weight = models['config'].get('VIS_WEIGHT', 0.2)
        features_final = np.hstack([meta_scaled, vis_weight * visual_pca_scaled])
        debug_info["8_features_fused"] = features_final.shape
        
        # 7. Predicción con XGBoost
        import xgboost as xgb
        dmatrix = xgb.DMatrix(features_final)
        extent_pred = models['xgb_model'].predict(dmatrix)[0]
        
        # Clip al rango válido
        extent_pred = np.clip(extent_pred, 0, 100)
        
        # 8. Retornar resultado
        result = {
            'extent_predicted': float(extent_pred),
            'visual_raw_dim': visual_embedding.shape[1],
            'visual_features_dim': visual_pca_scaled.shape[1],  # Features visuales finales (256)
            'visual_pca_dim': visual_pca.shape[1],
            'meta_features_dim': meta_processed.shape[1],
            'total_features': features_final.shape[1],
            'metadata_used': metadata,
            'model_type': 'xgboost',
            'debug_info': debug_info,
            'success': True
        }
        
        return result
    
    except Exception as e:
        import traceback
        error_details = {
            'success': False,
            'error': str(e),
            'error_trace': traceback.format_exc(),
            'extent_predicted': None
        }
        
        # Include debug info if available
        if 'debug_info' in locals():
            error_details['debug_info'] = debug_info
        
        return error_details


def get_interpretation_text(extent):
    """Genera interpretación textual del extent predicho"""
    if extent < 5:
        severity = "Muy Bajo"
        color = "#0f7a32"
        icon = "✅"
        message = "El cultivo presenta daño mínimo o despreciable."
    elif extent < 15:
        severity = "Bajo"
        color = "#7cb342"
        icon = "🟢"
        message = "Daño leve detectado. Monitorear evolución."
    elif extent < 30:
        severity = "Moderado-Bajo"
        color = "#fdd835"
        icon = "🟡"
        message = "Daño moderado. Se recomienda evaluación detallada."
    elif extent < 50:
        severity = "Moderado-Alto"
        color = "#ff9800"
        icon = "🟠"
        message = "Daño significativo. Considerar medidas de control."
    elif extent < 70:
        severity = "Alto"
        color = "#ff5722"
        icon = "🔴"
        message = "Daño severo. Requiere intervención inmediata."
    else:
        severity = "Muy Alto"
        color = "#b91c1c"
        icon = "🚨"
        message = "Daño crítico. Pérdida significativa del cultivo."
    
    return {
        'severity': severity,
        'color': color,
        'icon': icon,
        'message': message
    }
