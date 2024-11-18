import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Configuración inicial
st.title("Gestor de Finanzas Personales")
st.write("Esta app fue elaborada por Kevin Guio.")

# Diccionario para categorías
categorias = ["Comida", "Transporte", "Vivienda", "Entretenimiento", "Salud", "Otros"]

# Sesión para almacenar datos
if "data" not in st.session_state:
    st.session_state.data = {
        "Presupuesto": [],
        "Ingresos": [],
        "Gastos": [],
        "Metas de ahorro": []
    }

# Función para agregar registros
def agregar_registro(tipo, categoria, cantidad, fecha):
    st.session_state.data[tipo].append({
        "Fecha": fecha,
        "Categoría": categoria,
        "Cantidad": cantidad
    })

# Función para generar reportes
def generar_reporte(tipo, rango):
    df = pd.DataFrame(st.session_state.data[tipo])
    
    if not df.empty:
        # Asegurarse de que la columna "Cantidad" sea numérica
        df["Cantidad"] = pd.to_numeric(df["Cantidad"], errors="coerce")
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
        
        # Filtrar por rango de fechas
        df_rango = df[df["Fecha"].between(rango[0], rango[1])]
        
        # Agrupar por categoría y sumar
        return df_rango.groupby("Categoría").sum(numeric_only=True)["Cantidad"]
    else:
        return pd.Series(dtype="float64")

# Sección para gestionar presupuestos, ingresos, gastos y metas
st.sidebar.header("Gestión de Finanzas")
tipo_registro = st.sidebar.selectbox("Selecciona el tipo de registro:", ["Presupuesto", "Ingresos", "Gastos", "Metas de ahorro"])

with st.sidebar.form("formulario_registro"):
    categoria = st.selectbox("Categoría:", categorias)
    cantidad = st.number_input("Cantidad:", min_value=0.0, format="%.2f")
    fecha = st.date_input("Fecha:", value=datetime.today())
    submit = st.form_submit_button("Agregar")

if submit:
    agregar_registro(tipo_registro, categoria, cantidad, fecha)
    st.success(f"{tipo_registro} agregado correctamente.")

# Mostrar datos actuales
st.header("Datos Registrados")
tab_presupuesto, tab_ingresos, tab_gastos, tab_ahorro = st.tabs(["Presupuesto", "Ingresos", "Gastos", "Metas de ahorro"])

for tipo, tab in zip(["Presupuesto", "Ingresos", "Gastos", "Metas de ahorro"], 
                     [tab_presupuesto, tab_ingresos, tab_gastos, tab_ahorro]):
    with tab:
        if st.session_state.data[tipo]:
            st.table(pd.DataFrame(st.session_state.data[tipo]))
        else:
            st.write(f"No hay datos en {tipo}.")

# Generar reportes
st.header("Reportes")
reporte_tipo = st.selectbox("Selecciona el tipo de reporte:", ["Semanal", "Mensual"])
fecha_inicio = st.date_input("Fecha de inicio:", value=datetime.today())
fecha_fin = st.date_input("Fecha de fin:", value=datetime.today())

if st.button("Generar reporte"):
    rango = [pd.Timestamp(fecha_inicio), pd.Timestamp(fecha_fin)]
    presupuesto = generar_reporte("Presupuesto", rango)
    real = generar_reporte("Gastos", rango)
    diferencia = presupuesto.subtract(real, fill_value=0)
    
    st.subheader("Reporte de Diferencias")
    st.write("Diferencia entre lo presupuestado y lo real (gastos):")
    st.dataframe(diferencia)

    # Gráfico
    st.subheader("Visualización de Diferencias")
    fig, ax = plt.subplots()
    diferencia.plot(kind="bar", ax=ax, color="skyblue", edgecolor="black")
    ax.set_ylabel("Diferencia ($)")
    ax.set_xlabel("Categorías")
    ax.set_title("Diferencias Presupuestado vs Real")
    st.pyplot(fig)
