import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Simulador de Ventas - Noviembre 2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stMetric label {
        color: #2c3e50 !important;
        font-weight: 600 !important;
        font-size: 16px !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #667eea !important;
        font-weight: 700 !important;
        font-size: 28px !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #34495e !important;
        font-weight: 600 !important;
    }
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 15px;
        font-size: 18px;
    }
    .stButton>button:hover {
        background-color: #764ba2;
    }
    h1, h2, h3 {
        color: white;
    }
    .dataframe {
        background-color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Funciones auxiliares
@st.cache_data
def cargar_datos():
    """Carga el dataframe de inferencia y el modelo entrenado"""
    try:
        df = pd.read_csv('../data/processed/inferencia_df_transformado.csv')
        df['fecha'] = pd.to_datetime(df['fecha'])
        return df
    except Exception as e:
        st.error(f"❌ Error al cargar los datos: {e}")
        return None

@st.cache_resource
def cargar_modelo():
    """Carga el modelo entrenado"""
    try:
        modelo = joblib.load('../models/modelo_final.joblib')
        return modelo
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        return None

def aplicar_ajustes(df_producto, ajuste_descuento, escenario_competencia):
    """Aplica los ajustes de descuento y competencia al dataframe del producto"""
    df_ajustado = df_producto.copy()
    
    # Ajustar precio de venta según descuento
    descuento_decimal = ajuste_descuento / 100
    df_ajustado['precio_venta'] = df_ajustado['precio_base'] * (1 + descuento_decimal)
    
    # Ajustar precios de competencia según escenario
    factor_competencia = {
        "Actual (0%)": 1.0,
        "Competencia -5%": 0.95,
        "Competencia +5%": 1.05
    }[escenario_competencia]
    
    df_ajustado['Amazon'] = df_ajustado['Amazon'] * factor_competencia
    df_ajustado['Decathlon'] = df_ajustado['Decathlon'] * factor_competencia
    df_ajustado['Deporvillage'] = df_ajustado['Deporvillage'] * factor_competencia
    
    # Recalcular precio_competencia como promedio
    df_ajustado['precio_competencia'] = df_ajustado[['Amazon', 'Decathlon', 'Deporvillage']].mean(axis=1)
    
    # Recalcular descuento_porcentaje y ratio_precio
    df_ajustado['descuento_porcentaje'] = (df_ajustado['precio_venta'] - df_ajustado['precio_base']) / df_ajustado['precio_base']
    df_ajustado['ratio_precio'] = df_ajustado['precio_venta'] / df_ajustado['precio_competencia']
    
    return df_ajustado

def predecir_recursivamente(df_producto, modelo):
    """Realiza predicciones recursivas día por día actualizando los lags"""
    df_pred = df_producto.copy()
    df_pred = df_pred.sort_values('fecha').reset_index(drop=True)
    
    predicciones = []
    
    # Obtener las columnas que el modelo espera
    feature_columns = modelo.feature_names_in_
    
    # Verificar que todas las columnas necesarias existen
    columnas_faltantes = set(feature_columns) - set(df_pred.columns)
    if columnas_faltantes:
        st.error(f"❌ Columnas faltantes en el dataframe: {columnas_faltantes}")
        return df_pred
    
    # Historial de predicciones para calcular media móvil
    historial_predicciones = []
    
    for i in range(len(df_pred)):
        # Preparar las features en el orden correcto
        X = df_pred.iloc[i:i+1][feature_columns]
        
        # Hacer predicción
        pred = modelo.predict(X)[0]
        pred = max(0, pred)  # No permitir predicciones negativas
        predicciones.append(pred)
        historial_predicciones.append(pred)
        
        # Actualizar lags para el siguiente día (si no es el último día)
        if i < len(df_pred) - 1:
            # Desplazar lags hacia la derecha
            for lag in range(7, 1, -1):
                col_actual = f'unidades_vendidas_lag_{lag}'
                col_anterior = f'unidades_vendidas_lag_{lag-1}'
                if col_actual in df_pred.columns and col_anterior in df_pred.columns:
                    df_pred.loc[i+1, col_actual] = df_pred.loc[i, col_anterior]
            
            # Actualizar lag_1 con la predicción actual
            if 'unidades_vendidas_lag_1' in df_pred.columns:
                df_pred.loc[i+1, 'unidades_vendidas_lag_1'] = pred
            
            # Actualizar media móvil de 7 días
            if len(historial_predicciones) >= 7:
                ma7 = np.mean(historial_predicciones[-7:])
            else:
                ma7 = np.mean(historial_predicciones)
            
            if 'unidades_vendidas_ma7' in df_pred.columns:
                df_pred.loc[i+1, 'unidades_vendidas_ma7'] = ma7
    
    df_pred['unidades_predichas'] = predicciones
    df_pred['ingresos_proyectados'] = df_pred['unidades_predichas'] * df_pred['precio_venta']
    
    return df_pred

def crear_grafico_prediccion(df_pred):
    """Crea el gráfico de predicción diaria con Black Friday destacado"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Configurar estilo seaborn
    sns.set_style("whitegrid")
    
    # Crear la línea de predicción
    dias = df_pred['dia_mes'].values
    unidades = df_pred['unidades_predichas'].values
    
    sns.lineplot(x=dias, y=unidades, marker='o', linewidth=2.5, 
                 color='#667eea', markersize=8, ax=ax)
    
    # Marcar el Black Friday (día 28)
    black_friday_idx = df_pred[df_pred['dia_mes'] == 28].index[0]
    black_friday_unidades = df_pred.loc[black_friday_idx, 'unidades_predichas']
    
    # Línea vertical en Black Friday
    ax.axvline(x=28, color='#ff4b4b', linestyle='--', linewidth=2, alpha=0.7)
    
    # Punto destacado en Black Friday
    ax.plot(28, black_friday_unidades, 'o', color='#ff4b4b', 
            markersize=15, markeredgewidth=2, markeredgecolor='white')
    
    # Anotación para Black Friday
    ax.annotate('🛍️ Black Friday', 
                xy=(28, black_friday_unidades), 
                xytext=(28, black_friday_unidades * 1.15),
                fontsize=12, fontweight='bold', color='#ff4b4b',
                ha='center',
                arrowprops=dict(arrowstyle='->', color='#ff4b4b', lw=2))
    
    # Configurar etiquetas y título
    ax.set_xlabel('Día de Noviembre', fontsize=12, fontweight='bold')
    ax.set_ylabel('Unidades Vendidas', fontsize=12, fontweight='bold')
    ax.set_title('Predicción de Ventas Diarias - Noviembre 2025', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Ajustar ejes
    ax.set_xticks(range(1, 31, 2))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def crear_tabla_detallada(df_pred):
    """Crea la tabla detallada con los resultados diarios"""
    # Crear tabla con las columnas relevantes
    tabla = df_pred[['fecha', 'nombre_dia', 'precio_venta', 'precio_competencia', 
                     'descuento_porcentaje', 'unidades_predichas', 'ingresos_proyectados']].copy()
    
    # Formatear columnas
    tabla['fecha'] = tabla['fecha'].dt.strftime('%d/%m/%Y')
    tabla['precio_venta'] = tabla['precio_venta'].apply(lambda x: f"{x:.2f}€")
    tabla['precio_competencia'] = tabla['precio_competencia'].apply(lambda x: f"{x:.2f}€")
    tabla['descuento_porcentaje'] = tabla['descuento_porcentaje'].apply(lambda x: f"{x*100:.1f}%")
    tabla['unidades_predichas'] = tabla['unidades_predichas'].apply(lambda x: f"{int(x)}")
    tabla['ingresos_proyectados'] = tabla['ingresos_proyectados'].apply(lambda x: f"{x:.2f}€")
    
    # Renombrar columnas
    tabla.columns = ['Fecha', 'Día', 'Precio Venta', 'Precio Competencia', 
                     'Descuento', 'Unidades', 'Ingresos']
    
    # Añadir emoji para Black Friday
    tabla['Día'] = tabla.apply(
        lambda row: f"🛍️ {row['Día']}" if '28/11/2025' in row['Fecha'] else row['Día'],
        axis=1
    )
    
    return tabla

# ============= INTERFAZ PRINCIPAL =============

# Cargar datos y modelo
df = cargar_datos()
modelo = cargar_modelo()

if df is None or modelo is None:
    st.stop()

# ============= SIDEBAR =============
with st.sidebar:
    st.title("🎛️ Controles de Simulación")
    st.markdown("---")
    
    # Selector de producto
    productos_disponibles = sorted(df['nombre'].unique())
    producto_seleccionado = st.selectbox(
        "📦 Seleccionar Producto",
        productos_disponibles,
        help="Elige el producto para simular ventas"
    )
    
    st.markdown("---")
    
    # Slider de descuento
    ajuste_descuento = st.slider(
        "💰 Ajuste de Descuento",
        min_value=-50,
        max_value=50,
        value=0,
        step=5,
        format="%d%%",
        help="Ajusta el descuento sobre el precio base"
    )
    
    st.markdown("---")
    
    # Selector de escenario de competencia
    st.markdown("**🏪 Escenario de Competencia**")
    escenario_competencia = st.radio(
        "",
        ["Actual (0%)", "Competencia -5%", "Competencia +5%"],
        help="Simula cambios en los precios de la competencia"
    )
    
    st.markdown("---")
    
    # Botón de simulación
    simular = st.button("🚀 Simular Ventas", use_container_width=True)

# ============= ZONA PRINCIPAL =============

# Header
st.title(f"📊 Dashboard de Simulación - Noviembre 2025")
st.markdown(f"### Producto: **{producto_seleccionado}**")
st.markdown("---")

# Procesar simulación
if simular:
    with st.spinner("⏳ Procesando predicciones recursivas..."):
        # Filtrar datos del producto
        df_producto = df[df['nombre'] == producto_seleccionado].copy()
        
        # Aplicar ajustes
        df_ajustado = aplicar_ajustes(df_producto, ajuste_descuento, escenario_competencia)
        
        # Hacer predicciones recursivas
        df_pred = predecir_recursivamente(df_ajustado, modelo)
        
        # Calcular KPIs
        unidades_totales = df_pred['unidades_predichas'].sum()
        ingresos_totales = df_pred['ingresos_proyectados'].sum()
        precio_promedio = df_pred['precio_venta'].mean()
        descuento_promedio = df_pred['descuento_porcentaje'].mean() * 100
        
        # Mostrar KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📦 Unidades Totales",
                value=f"{int(unidades_totales):,}",
                delta=None
            )
        
        with col2:
            st.metric(
                label="💶 Ingresos Proyectados",
                value=f"{ingresos_totales:,.2f}€",
                delta=None
            )
        
        with col3:
            st.metric(
                label="💵 Precio Promedio",
                value=f"{precio_promedio:.2f}€",
                delta=None
            )
        
        with col4:
            st.metric(
                label="🏷️ Descuento Promedio",
                value=f"{descuento_promedio:.1f}%",
                delta=None
            )
        
        st.markdown("---")
        
        # Gráfico de predicción
        st.subheader("📈 Predicción de Ventas Diarias")
        fig = crear_grafico_prediccion(df_pred)
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Tabla detallada
        st.subheader("📋 Detalle Diario de Predicciones")
        tabla = crear_tabla_detallada(df_pred)
        st.dataframe(tabla, use_container_width=True, height=400)
        
        st.markdown("---")
        
        # Comparativa de escenarios
        st.subheader("🔍 Comparativa de Escenarios de Competencia")
        
        with st.spinner("⏳ Calculando escenarios alternativos..."):
            escenarios_resultados = {}
            
            for escenario in ["Actual (0%)", "Competencia -5%", "Competencia +5%"]:
                df_esc = aplicar_ajustes(df_producto, ajuste_descuento, escenario)
                df_esc_pred = predecir_recursivamente(df_esc, modelo)
                
                escenarios_resultados[escenario] = {
                    'unidades': df_esc_pred['unidades_predichas'].sum(),
                    'ingresos': df_esc_pred['ingresos_proyectados'].sum()
                }
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🏪 Actual (0%)**")
                st.metric(
                    "Unidades",
                    f"{int(escenarios_resultados['Actual (0%)']['unidades']):,}"
                )
                st.metric(
                    "Ingresos",
                    f"{escenarios_resultados['Actual (0%)']['ingresos']:,.2f}€"
                )
            
            with col2:
                st.markdown("**📉 Competencia -5%**")
                delta_unidades = escenarios_resultados['Competencia -5%']['unidades'] - escenarios_resultados['Actual (0%)']['unidades']
                delta_ingresos = escenarios_resultados['Competencia -5%']['ingresos'] - escenarios_resultados['Actual (0%)']['ingresos']
                
                st.metric(
                    "Unidades",
                    f"{int(escenarios_resultados['Competencia -5%']['unidades']):,}",
                    delta=f"{int(delta_unidades):,}"
                )
                st.metric(
                    "Ingresos",
                    f"{escenarios_resultados['Competencia -5%']['ingresos']:,.2f}€",
                    delta=f"{delta_ingresos:,.2f}€"
                )
            
            with col3:
                st.markdown("**📈 Competencia +5%**")
                delta_unidades = escenarios_resultados['Competencia +5%']['unidades'] - escenarios_resultados['Actual (0%)']['unidades']
                delta_ingresos = escenarios_resultados['Competencia +5%']['ingresos'] - escenarios_resultados['Actual (0%)']['ingresos']
                
                st.metric(
                    "Unidades",
                    f"{int(escenarios_resultados['Competencia +5%']['unidades']):,}",
                    delta=f"{int(delta_unidades):,}"
                )
                st.metric(
                    "Ingresos",
                    f"{escenarios_resultados['Competencia +5%']['ingresos']:,.2f}€",
                    delta=f"{delta_ingresos:,.2f}€"
                )
        
        st.success("✅ Simulación completada exitosamente")

else:
    # Mensaje inicial
    st.info("👈 Configura los parámetros en el panel lateral y presiona **'Simular Ventas'** para comenzar")
    
    # Mostrar información del producto seleccionado
    df_producto_info = df[df['nombre'] == producto_seleccionado].iloc[0]
    
    st.markdown("### 📌 Información del Producto Seleccionado")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Categoría:**")
        st.write(df_producto_info['categoria'])
        
    with col2:
        st.markdown("**Subcategoría:**")
        st.write(df_producto_info['subcategoria'])
        
    with col3:
        st.markdown("**Precio Base:**")
        st.write(f"{df_producto_info['precio_base']:.2f}€")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: white;'>"
    "🤖 Desarrollado con Streamlit | Marco Gomez | 2025"
    "</div>",
    unsafe_allow_html=True
)
