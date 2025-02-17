# ... (todo el código anterior se mantiene igual)

def load_data(uploaded_file=None, url=None):
    """Carga y normaliza datos desde CSV con manejo robusto de columnas."""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        elif url:
            df = pd.read_csv(url)
        else:
            raise ValueError("Debe proporcionar un archivo o URL")
        
        # Normalización mejorada de nombres de columnas
        df.columns = [unidecode(col).strip().upper()
                      .replace(' ', '_')
                      .replace('MUNICIPALIDAD', 'MUNICIPIO')
                      .replace('CIUDAD', 'MUNICIPIO')
                      .replace('LOCALIDAD', 'MUNICIPIO')
                      for col in df.columns]
        
        # Verificar y corregir nombres esenciales
        column_mapping = {
            'DEPARTAMENTO': 'DPTO',
            'PROVINCIA': 'DPTO',
            'MUNICIPIO': 'MUNICIPIO',
            'VOLUMEN': 'VOLUMEN_M3',
            'MADERA': 'ESPECIE'
        }
        
        for original, nuevo in column_mapping.items():
            if original in df.columns and nuevo not in df.columns:
                df = df.rename(columns={original: nuevo})
        
        # Validar columnas requeridas
        required_columns = {'DPTO', 'VOLUMEN_M3'}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Columnas obligatorias faltantes: {', '.join(missing)}")
            
        return df.copy()
    
    except Exception as e:
        raise Exception(f"Error cargando datos: {str(e)}") from e

def main():
    st.title("🌳 Análisis de Producción Maderera")
    
    uploaded_file = st.file_uploader("Subir archivo CSV", type="csv")
    url = st.text_input("O ingresar URL de datos CSV")
    
    if uploaded_file or url:
        try:
            # Cargar y procesar datos
            df = load_data(uploaded_file, url)
            df_clean = handle_missing_values(df)
            
            # Verificar existencia de columnas municipales
            has_municipio = 'MUNICIPIO' in df_clean.columns
            disable_municipal = not has_municipio
            
            # Mostrar secciones condicionales
            if not has_municipio:
                st.warning("⚠️ El dataset no contiene información municipal. Algunas funcionalidades estarán limitadas.")
            
            # Sección de Top 5 por Departamento
            st.header("📊 Análisis por Departamento")
            try:
                top_species = get_top_species(df_clean)
                st.dataframe(top_species)
            except ValueError as e:
                st.error(str(e))
            
            # Sección de Top 10 Especies
            st.header("🏆 Top 10 Especies a Nivel Nacional")
            try:
                fig_top10 = plot_top_species(df_clean)
                st.plotly_chart(fig_top10, use_container_width=True)
                
                # Calcular métricas
                top_data = df_clean.groupby('ESPECIE')['VOLUMEN_M3'].sum().nlargest(10)
                cols = st.columns(3)
                cols[0].metric("📦 Volumen Total Top 10", f"{top_data.sum():,.0f} m³")
                cols[1].metric("📊 Promedio por Especie", f"{top_data.mean():,.0f} m³")
                cols[2].metric("🌿 Especies Únicas", len(top_data))
                
            except ValueError as e:
                st.error(str(e))
            
            # Sección de Municipios (solo si hay datos)
            if not disable_municipal:
                st.header("🏙️ Top 10 Municipios")
                try:
                    top_municipios = get_top_municipalities(df_clean)
                    
                    cols = st.columns(3)
                    cols[0].metric("📦 Volumen Total", f"{top_municipios['VOLUMEN_TOTAL'].sum():,.0f} m³")
                    cols[1].metric("📍 Municipios Únicos", top_municipios['MUNICIPIO'].nunique())
                    cols[2].metric("🗺️ Departamentos", top_municipios['DPTO'].nunique())
                    
                    st.dataframe(top_municipios.sort_values('VOLUMEN_TOTAL', ascending=False))
                    fig_municipios = plot_municipality_map(top_municipios)
                    st.plotly_chart(fig_municipios, use_container_width=True)
                    
                except ValueError as e:
                    st.error(str(e))
            
            # Resto de secciones...
            
        except Exception as e:
            st.error(f"🚨 Error general: {str(e)}")

# ... (resto del código se mantiene igual)
