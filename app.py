# eda_page.py - Dashboard con diseño agrícola y accesibilidad mejorada

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============== PALETA DE COLORES AGRÍCOLA (W3C AAA Compliant) ==============

COLORS = {
    "primary": "#1e3a0f",      # Verde muy oscuro (AAA contrast)
    "secondary": "#0d1f04",    # Verde extremadamente oscuro
    "accent": "#c86200",       # Naranja oscuro (mejor contraste)
    "highlight": "#d64500",    # Naranja rojizo
    "background": "#fefae0",   # Crema claro
    "card_bg": "#ffffff",      # Blanco para tarjetas
    "card_border": "#b8c5a0",  # Verde suave para bordes
    "text": "#0d1f04",         # Verde muy oscuro (AAA)
    "text_light": "#2d5016",   # Verde oscuro para secundario
    "success": "#0f7a32",      # Verde oscuro
    "warning": "#b91c1c",      # Rojo oscuro
}

CHART_COLORS = ["#1e3a0f", "#c86200", "#d64500", "#0f7a32", "#2d5016", "#b91c1c"]

# ============== CSS PERSONALIZADO MEJORADO ==============

def apply_custom_css():
    st.markdown(f"""
        <style>
        /* Fuentes y tema general */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: {COLORS['text']};
        }}
        
        /* Fondo principal */
        .stApp {{
            background: {COLORS['background']};
        }}
        [data-testid="stHeader"] {{
            background: {COLORS['background']}; /* Mismo color que el fondo */
            box-shadow: none !important;       /* Quitar sombra */
            border-bottom: none !important; /* Quitar borde */
        }}
        /* Mejora de contraste en títulos */
        h1, h2, h3, h4, h5, h6 {{
            color: {COLORS['primary']} !important;
            font-weight: 700 !important;
            animation: fadeInDown 0.5s ease-out;
        }}
        
        h1 {{
            font-size: 2.25rem !important;
        }}
        
        h2 {{
            font-size: 1.75rem !important;
        }}
        
        h3 {{
            font-size: 1.5rem !important;
        }}
        
        h4 {{
            font-size: 1.25rem !important;
        }}
        
        @keyframes fadeInDown {{
            from {{
                opacity: 0;
                transform: translateY(-15px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        /* Fix para métricas de Streamlit - OCULTAR divs internos */
        [data-testid="stMetric"] {{
            background-color: {COLORS['card_bg']};
            padding: 1.25rem;
            border-radius: 12px;
            border: 2px solid {COLORS['card_border']};
            box-shadow: 0 2px 8px rgba(13, 31, 4, 0.08);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        [data-testid="stMetric"]:hover {{
            transform: translateY(-4px);
            box-shadow: 0 6px 20px rgba(13, 31, 4, 0.12);
            border-color: {COLORS['primary']};
        }}
        
        [data-testid="stMetricValue"] {{
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: {COLORS['primary']} !important;
        }}
        
        [data-testid="stMetricLabel"] {{
            color: {COLORS['text']} !important;
            font-size: 0.875rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        

        [data-testid="stMetric"] > div > div:first-child {{
            display: block !important;
        }}
        
        /* Selectboxes mejorados */
        .stSelectbox label {{
            color: {COLORS['primary']} !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
        }}
        
        .stSelectbox > div > div {{
            background-color: {COLORS['card_bg']} !important;
            border: 2px solid {COLORS['primary']} !important;
            border-radius: 8px !important;
            transition: all 0.2s ease !important;
        }}
        
        .stSelectbox > div > div:hover {{
            border-color: {COLORS['accent']} !important;
            box-shadow: 0 2px 8px rgba(200, 98, 0, 0.15) !important;
        }}
        
        .stSelectbox [data-baseweb="select"] > div {{
            color: {COLORS['text']} !important;
            font-weight: 500 !important;
        }}
        
        /* Tabs mejorados con mejor contraste */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 6px;
            background-color: transparent;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background-color: {COLORS['card_bg']};
            border-radius: 8px 8px 0 0;
            padding: 0.75rem 1.5rem;
            border: 2px solid {COLORS['card_border']};
            color: {COLORS['text']};
            font-weight: 600;
            font-size: 0.95rem;
            transition: all 0.2s ease;
        }}
        
        .stTabs [data-baseweb="tab"]:hover {{
            background-color: {COLORS['background']};
            border-color: {COLORS['primary']};
            transform: translateY(-2px);
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: {COLORS['primary']} !important;
            color: white !important;
            border-color: {COLORS['primary']} !important;
            font-weight: 700;
        }}
        
        /* Panel de contenido de tabs */
        .stTabs [data-baseweb="tab-panel"] {{
            padding-top: 1.5rem;
        }}
        
        /* Dataframes */
        .dataframe {{
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(13, 31, 4, 0.08);
            border: 1px solid {COLORS['card_border']};
        }}
        
        /* Scrollbar mejorado */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: {COLORS['background']};
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: {COLORS['primary']};
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: {COLORS['secondary']};
        }}
        
        /* Sidebar mejorado */
        section[data-testid="stSidebar"] {{
            background-color: {COLORS['primary']};
            border-right: 2px solid {COLORS['accent']};
        }}
        
        section[data-testid="stSidebar"] h2 {{
            color: #ffffff !important;
            font-weight: 800 !important;
            margin-bottom: 1.5rem !important;
        }}
        
        section[data-testid="stSidebar"] label {{
            color: #ffffff !important;
            font-weight: 600 !important;
        }}
        
        section[data-testid="stSidebar"] .stMarkdown {{
            color: #2d5016 !important;
        }}
        
        section[data-testid="stSidebar"] .stMarkdown p {{
            color: #2d5016 !important;
        }}
        
        section[data-testid="stSidebar"] [data-baseweb="select"] > div {{
            background-color: rgba(255, 255, 255, 0.15) !important;
            color: #ffffff !important;
        }}
        
        section[data-testid="stSidebar"] [data-baseweb="select"] span {{
            color: #ffffff !important;
        }}
        
        /* Alertas y warnings */
        .stAlert {{
            border-radius: 8px;
            border-left: 4px solid {COLORS['accent']};
        }}
        
        /* Mejora de accesibilidad para focus */
        button:focus, 
        [data-baseweb="select"]:focus,
        input:focus {{
            outline: 3px solid {COLORS['accent']} !important;
            outline-offset: 2px !important;
        }}
        </style>
    """, unsafe_allow_html=True)


def apply_plotly_theme(fig, title_text=""):
    """Aplica tema consistente y accesible a gráficas de Plotly"""
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="rgba(255, 255, 255, 0.98)",
        paper_bgcolor="rgba(255, 255, 255, 0.98)",
        font=dict(
            color=COLORS['text'],
            family="Inter, sans-serif",
            size=13
        ),
        title=dict(
            text=title_text,
            font=dict(size=18, color=COLORS['primary'], family="Inter", weight=700),
            x=0.5,
            xanchor='center',
            y=0.98,
            yanchor='top'
        ),
        margin=dict(l=70, r=50, t=70, b=60),
        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            font_family="Inter",
            font_color=COLORS['text'],
            bordercolor=COLORS['primary']
        ),
        transition_duration=400,
        xaxis=dict(
            gridcolor='rgba(30, 58, 15, 0.12)',
            zeroline=False,
            title_font=dict(color=COLORS['text'], size=13, family="Inter", weight=600),
            tickfont=dict(color=COLORS['text'], size=12)
        ),
        yaxis=dict(
            gridcolor='rgba(30, 58, 15, 0.12)',
            zeroline=False,
            title_font=dict(color=COLORS['text'], size=13, family="Inter", weight=600),
            tickfont=dict(color=COLORS['text'], size=12)
        )
    )
    return fig


# ============== CARGA DE DATOS ==============

@st.cache_data
def load_train_data(train_path="Train.csv", test_path="Test.csv"):
    """Carga Train.csv y, si existe, Test.csv; concatena ambos (añade columna 'split')."""
    df_train = pd.read_csv(train_path)
    df_train["split"] = "train"

    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path)
        df_test["split"] = "test"
        df = pd.concat([df_train, df_test], ignore_index=True, sort=False)
    else:
        df = df_train.copy()

    # Normalizar extent
    if "extent" in df.columns:
        df["extent"] = pd.to_numeric(df["extent"], errors="coerce").fillna(0.0)
        df["extent"] = df["extent"].clip(0, 100)

    # Limpiar columnas categóricas
    for col in ["damage", "season", "growth_stage"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            # Restaurar NaN para valores que se convirtieron a 'nan' strings
            df.loc[df[col].isin(["nan", "NaN", "None"]), col] = np.nan

    return df.reset_index(drop=True)


@st.cache_data
def load_model_metrics():
    data = [
        ["Metadatos solo (RF)", 3.11, 9.19, 0.76],
        ["Visual solo (XGB)", 12.30, 19.24, -0.07],
        ["Early Fusion inicial (XGB)", 3.29, 9.38, 0.74],
        ["Early Fusion PCA + peso", 3.32, 9.38, 0.75],
        ["Early Fusion tunable (XGB)", 2.56, 7.26, 0.85],
        ["Residual RF + Visual", 2.77, 8.33, 0.80],
    ]
    df = pd.DataFrame(data, columns=["Modelo", "MAE", "RMSE", "R2"])
    return df


# ============== COMPONENTES VISUALES ==============

def create_comparison_chart(metrics_df):
    """Crea gráfica de comparación de modelos con subplots"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("MAE (menor es mejor)", "RMSE (menor es mejor)", "R² Score (mayor es mejor)"),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    # MAE
    fig.add_trace(
        go.Bar(
            x=metrics_df["Modelo"],
            y=metrics_df["MAE"],
            marker=dict(
                color=metrics_df["MAE"],
                colorscale=[[0, COLORS['success']], [1, COLORS['warning']]],
                showscale=False,
                line=dict(color=COLORS['primary'], width=1)
            ),
            name="MAE",
            hovertemplate="<b>%{x}</b><br>MAE: %{y:.2f}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # RMSE
    fig.add_trace(
        go.Bar(
            x=metrics_df["Modelo"],
            y=metrics_df["RMSE"],
            marker=dict(
                color=metrics_df["RMSE"],
                colorscale=[[0, COLORS['success']], [1, COLORS['warning']]],
                showscale=False,
                line=dict(color=COLORS['primary'], width=1)
            ),
            name="RMSE",
            hovertemplate="<b>%{x}</b><br>RMSE: %{y:.2f}<extra></extra>"
        ),
        row=1, col=2
    )
    
    # R2
    fig.add_trace(
        go.Bar(
            x=metrics_df["Modelo"],
            y=metrics_df["R2"],
            marker=dict(
                color=metrics_df["R2"],
                colorscale=[[0, COLORS['warning']], [1, COLORS['success']]],
                showscale=False,
                line=dict(color=COLORS['primary'], width=1)
            ),
            name="R²",
            hovertemplate="<b>%{x}</b><br>R²: %{y:.2f}<extra></extra>"
        ),
        row=1, col=3
    )
    
    fig.update_xaxes(
        tickangle=45, 
        title_font=dict(color=COLORS['text'], size=12, family="Inter"),
        tickfont=dict(color=COLORS['text'], size=11, family="Inter")
    )
    fig.update_yaxes(
        title_font=dict(color=COLORS['text'], size=12, family="Inter"),
        tickfont=dict(color=COLORS['text'], size=11, family="Inter")
    )
    fig.update_layout(
        height=450,
        showlegend=False,
        font=dict(family="Inter", color=COLORS['text'], size=12),
        plot_bgcolor="rgba(255, 255, 255, 0.98)",
        paper_bgcolor="rgba(255, 255, 255, 0.98)",
        annotations=[
            dict(
                font=dict(size=13, color=COLORS['primary'], family="Inter", weight=600),
            ) for _ in range(3)
        ]
    )
    
    return fig


def create_info_card(title, content_list):
    """Crea una tarjeta informativa accesible"""
    items_html = "".join([f"<li><strong>{k}:</strong> {v}</li>" for k, v in content_list])
    
    return f"""
        <div style='background: {COLORS["card_bg"]}; 
                    padding: 1.25rem; 
                    border-radius: 12px; 
                    border: 2px solid {COLORS["card_border"]};
                    box-shadow: 0 2px 8px rgba(13, 31, 4, 0.08);
                    height: 100%;'>
            <h4 style='color: {COLORS["primary"]}; 
                       margin-top: 0; 
                       margin-bottom: 1rem;
                       font-weight: 700;
                       font-size: 1.1rem;'>
                {title}
            </h4>
            <ul style='color: {COLORS["text"]}; 
                       line-height: 1.8; 
                       font-size: 0.95rem;
                       margin: 0;
                       padding-left: 1.25rem;'>
                {items_html}
            </ul>
        </div>
    """


# ============== PÁGINA PRINCIPAL ==============

def main():
    st.set_page_config(
        page_title="🌾 Dashboard de Análisis de Cultivos",
        layout="wide",
        page_icon="🌾",
        initial_sidebar_state="expanded"
    )
    
    apply_custom_css()
    
    # Header mejorado con mejor accesibilidad
    st.markdown(f"""
        <header role="banner" style='text-align: center; padding: 1.5rem 1rem; margin-bottom: 1rem;'>
            <h1 style='font-size: 2.25rem; 
                       color: {COLORS["primary"]}; 
                       margin-bottom: 0.5rem; 
                       font-weight: 800;
                       line-height: 1.2;'>
                🌾 Dashboard de Análisis de Daños en Cultivos
            </h1>
            <p style='color: {COLORS["text"]}; 
                      font-size: 1.1rem; 
                      font-weight: 500;
                      margin: 0;'>
                Exploración de datos y comparación de modelos multimodales
            </p>
        </header>
    """, unsafe_allow_html=True)
    
    # Carga de datos
    train_path = "Train.csv"
    if not os.path.exists(train_path):
        st.error("⚠️ No se encontró Train.csv. Verifica la ruta del archivo.")
        return
    
    df = load_train_data(train_path)
    metrics_df = load_model_metrics()
    
    # Sidebar con filtros
    with st.sidebar:
        st.markdown(f"""
            <h2 style='color: {COLORS["primary"]}; 
                       font-weight: 800; 
                       font-size: 1.5rem; 
                       margin-bottom: 1.5rem;'>
                🔍 Filtros de Datos
            </h2>
        """, unsafe_allow_html=True)
        
        damage_values = ["(Todos)"] + sorted(df["damage"].dropna().unique().tolist())
        season_values = ["(Todos)"] + sorted(df["season"].dropna().unique().tolist())
        stage_values = ["(Todos)"] + sorted(df["growth_stage"].dropna().unique().tolist())
        
        sel_damage = st.selectbox("🦠 Tipo de daño", damage_values, key="damage_filter")
        sel_season = st.selectbox("🌤️ Temporada", season_values, key="season_filter")
        sel_stage = st.selectbox("🌱 Etapa fenológica", stage_values, key="stage_filter")
        
        st.markdown("---")
        st.markdown(f"""
            <div style='text-align: center; 
                        padding: 1rem; 
                        background-color: {COLORS["card_bg"]}; 
                        border-radius: 8px; 
                        margin-top: 1.5rem; 
                        border: 2px solid {COLORS["card_border"]};
                        box-shadow: 0 2px 8px rgba(13, 31, 4, 0.08);'>
                <p style='color: {COLORS["text"]}; 
                          font-size: 0.9rem; 
                          margin: 0; 
                          font-weight: 600;
                          line-height: 1.5;'>
                    <strong>Proyecto 2</strong><br>
                    Data Science<br>
                    Universidad del Valle
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Aplicar filtros
    df_filt = df.copy()
    if sel_damage != "(Todos)":
        df_filt = df_filt[df_filt["damage"] == sel_damage]
    if sel_season != "(Todos)":
        df_filt = df_filt[df_filt["season"] == sel_season]
    if sel_stage != "(Todos)":
        df_filt = df_filt[df_filt["growth_stage"] == sel_stage]
    
    if len(df_filt) == 0:
        st.warning("⚠️ No hay registros que coincidan con los filtros seleccionados. Intenta ajustar los criterios.")
        return
    
    # KPIs principales usando métricas nativas de Streamlit
    st.markdown("<div role='region' aria-label='Métricas principales'>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 REGISTROS",
            value=f"{len(df_filt):,}"
        )
    
    with col2:
        st.metric(
            label="📈 EXTENT PROMEDIO",
            value=f"{df_filt['extent'].mean():.1f}%"
        )
    
    with col3:
        st.metric(
            label="⚠️ EXTENT MÁXIMO",
            value=f"{df_filt['extent'].max():.1f}%"
        )
    
    with col4:
        st.metric(
            label="📉 DESVIACIÓN EST.",
            value=f"{df_filt['extent'].std():.1f}%"
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs principales
    tab1, tab2, tab3 = st.tabs([
        "📊 Distribuciones", 
        "🔬 Análisis Categórico", 
        "🤖 Comparación de Modelos"
    ])
    
    with tab1:
        col1, col2 = st.columns([2.5, 1.5])
        
        with col1:
            fig_hist = px.histogram(
                df_filt,
                x="extent",
                nbins=40,
                marginal="box",
                color_discrete_sequence=[COLORS['primary']],
                labels={"extent": "Extent de Daño (%)", "count": "Frecuencia"}
            )
            fig_hist = apply_plotly_theme(fig_hist, "Distribución de Extent de Daño")
            fig_hist.update_traces(
                marker_line_color=COLORS['secondary'], 
                marker_line_width=1.5,
                opacity=0.9
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            insights_data = [
                ("Media", f"{df_filt['extent'].mean():.1f}%"),
                ("Mediana", f"{df_filt['extent'].median():.1f}%"),
                ("Percentil 25", f"{df_filt['extent'].quantile(0.25):.1f}%"),
                ("Percentil 75", f"{df_filt['extent'].quantile(0.75):.1f}%"),
                ("Rango", f"{df_filt['extent'].max() - df_filt['extent'].min():.1f}%")
            ]
            st.markdown(
                create_info_card("📌 Estadísticas Descriptivas", insights_data),
                unsafe_allow_html=True
            )
    
    with tab2:
        st.markdown("### 🎻 Análisis por Variables Categóricas")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_season = px.violin(
                df_filt,
                x="season",
                y="extent",
                color="season",
                box=True,
                color_discrete_sequence=CHART_COLORS,
                labels={
                    "extent": "Extent de Daño (%)",
                    "season": "Temporada"
                }
            )
            fig_season = apply_plotly_theme(fig_season, "Extent por Temporada")
            fig_season.update_layout(showlegend=False)
            st.plotly_chart(fig_season, use_container_width=True)
        
        with col2:
            fig_damage = px.box(
                df_filt,
                x="damage",
                y="extent",
                color="damage",
                color_discrete_sequence=CHART_COLORS,
                labels={
                    "extent": "Extent de Daño (%)",
                    "damage": "Tipo de Daño"
                }
            )
            fig_damage = apply_plotly_theme(fig_damage, "Extent por Tipo de Daño")
            fig_damage.update_xaxes(tickangle=45)
            fig_damage.update_layout(showlegend=False)
            st.plotly_chart(fig_damage, use_container_width=True)
    
    with tab3:
        st.markdown("### 🤖 Rendimiento de Modelos Multimodales")
        
        col1, col2 = st.columns([1.2, 2])
        
        with col1:
            st.markdown("#### 📋 Métricas de Validación")
            styled_df = metrics_df.style.background_gradient(
                subset=['MAE', 'RMSE'],
                cmap='RdYlGn_r'
            ).background_gradient(
                subset=['R2'],
                cmap='RdYlGn'
            ).format({
                'MAE': '{:.2f}',
                'RMSE': '{:.2f}',
                'R2': '{:.2f}'
            }).set_table_styles([
                {'selector': 'th', 'props': [('background-color', COLORS['primary']), 
                                              ('color', 'white'), 
                                              ('font-weight', '700'),
                                              ('text-align', 'center')]},
                {'selector': 'td', 'props': [('text-align', 'center'),
                                              ('padding', '8px')]}
            ])
            st.dataframe(styled_df, use_container_width=True, height=300)
        
        with col2:
            fig_comparison = create_comparison_chart(metrics_df)
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        best_model = metrics_df.loc[metrics_df['R2'].idxmax()]
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, {COLORS["primary"]}, {COLORS["accent"]}); 
                        padding: 30px; border-radius: 16px; color: white; text-align: center; 
                        box-shadow: 0 8px 32px rgba(30, 58, 15, 0.4); margin-top: 25px;
                        border: 2px solid {COLORS["secondary"]};'>
                <h3 style='margin: 0 0 15px 0; color: white; font-weight: 700; font-size: 1.5rem;'>
                    🏆 Mejor Modelo
                </h3>
                <h2 style='margin: 10px 0 20px 0; color: white; font-weight: 800; font-size: 1.8rem;'>
                    {best_model['Modelo']}
                </h2>
                <div style='display: flex; justify-content: center; gap: 40px; margin-top: 20px;'>
                    <div style='background: rgba(255,255,255,0.15); padding: 15px 25px; border-radius: 12px;'>
                        <div style='font-size: 0.85rem; opacity: 0.9; margin-bottom: 5px;'>MAE</div>
                        <div style='font-size: 1.75rem; font-weight: 800;'>{best_model['MAE']:.2f}</div>
                    </div>
                    <div style='background: rgba(255,255,255,0.15); padding: 15px 25px; border-radius: 12px;'>
                        <div style='font-size: 0.85rem; opacity: 0.9; margin-bottom: 5px;'>RMSE</div>
                        <div style='font-size: 1.75rem; font-weight: 800;'>{best_model['RMSE']:.2f}</div>
                    </div>
                    <div style='background: rgba(255,255,255,0.15); padding: 15px 25px; border-radius: 12px;'>
                        <div style='font-size: 0.85rem; opacity: 0.9; margin-bottom: 5px;'>R²</div>
                        <div style='font-size: 1.75rem; font-weight: 800;'>{best_model['R2']:.2f}</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()