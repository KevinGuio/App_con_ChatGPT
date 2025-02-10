def plot_cluster_on_basemap(df):
    """Plot cluster analysis results on the base map (South America) with clusters colored.

    Args:
        df (pd.DataFrame): DataFrame with filled data.
    """
    # Realiza el análisis de clúster (la función cluster_analysis ya retorna y muestra una tabla)
    df_cluster = cluster_analysis(df)
    # Cargar el mapa base filtrado a Sudamérica
    base_map = cargar_mapa_base(URL_GEOPACKAGE)
    if base_map is None:
        st.error("No se pudo cargar el mapa base.")
        return
    # Convertir los datos de clúster a GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df_cluster,
        geometry=gpd.points_from_xy(df_cluster['Longitud'], df_cluster['Latitud']),
        crs="EPSG:4326"
    )
    # Crear la figura y graficar
    fig, ax = plt.subplots(figsize=(10, 6))
    base_map.plot(ax=ax, color="lightgray", edgecolor="black")
    gdf.plot(ax=ax, column="cluster", cmap="viridis", markersize=50, legend=True)
    ax.set_title("Mapa de Clústeres basado en Frecuencia de Compra")
    st.pyplot(fig)
