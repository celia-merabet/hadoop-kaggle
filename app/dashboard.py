import streamlit as st
import plotly.express as px
from pyspark.sql import SparkSession

st.set_page_config(page_title="Olist - Analyse des livraisons", layout="wide")
st.title("Analyse Big Data des livraisons e-commerce – Olist")
st.caption("Architecture Hadoop HDFS · PySpark · Delta Lake · Streamlit")

@st.cache_resource
def get_spark():
    return (
        SparkSession.builder
        .appName("OlistStreamlitDashboard")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

@st.cache_data(ttl=60)
def load_gold_table(table_name):
    return get_spark().read.format("delta").load(
        f"/lakehouse/gold/{table_name}"
    ).toPandas()

try:
    category = load_gold_table("category_performance")
    monthly = load_gold_table("monthly_delivery")
    states = load_gold_table("late_orders_by_state")
    segments = load_gold_table("customer_segments")

    st.success("Tables Gold chargées depuis HDFS / Delta Lake.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Commandes analysées", f"{int(category.total_orders.sum()):,}")
    c2.metric("Catégories produits", f"{category.product_category.nunique():,}")
    c3.metric("États présents", f"{states.customer_state.nunique():,}")

    st.subheader("1. Performance par catégorie de produit")
    selected_categories = st.multiselect(
        "Filtrer les catégories",
        sorted(category["product_category"].dropna().unique()),
        default=sorted(category["product_category"].dropna().unique())[:10]
    )
    filtered = category[category["product_category"].isin(selected_categories)]
    st.plotly_chart(
        px.bar(filtered.sort_values("total_orders", ascending=False),
               x="product_category", y="total_orders",
               title="Nombre de commandes par catégorie"),
        use_container_width=True
    )
    with st.expander("Requête SQL et justification"):
        st.code("""
SELECT product_category, COUNT(*) AS total_orders
FROM v_silver
GROUP BY product_category
ORDER BY total_orders DESC
""", language="sql")
        st.write("Le nombre de commandes permet d'identifier les catégories les plus représentées.")

    st.subheader("2. Évolution mensuelle des délais")
    st.plotly_chart(
        px.line(monthly, x="purchase_month", y="average_delivery_days",
                markers=True, title="Délai moyen de livraison par mois"),
        use_container_width=True
    )
    with st.expander("Requête SQL et justification"):
        st.code("SELECT purchase_month, AVG(delivery_days) FROM v_silver GROUP BY purchase_month", language="sql")
        st.write("La fonction LAG compare chaque mois à la période précédente.")

    st.subheader("3. Retards par État brésilien")
    st.plotly_chart(
        px.bar(states.sort_values("average_delay_days", ascending=False),
               x="customer_state", y="average_delay_days",
               title="Retard moyen par État"),
        use_container_width=True
    )
    with st.expander("Requête SQL et justification"):
        st.code("SELECT customer_state, AVG(delay_days) FROM v_silver GROUP BY customer_state", language="sql")
        st.write("La moyenne est accompagnée d'une analyse de distribution dans les notebooks ; les valeurs extrêmes peuvent l'influencer.")

    st.subheader("4. Segmentation des clients")
    segment_counts = segments.groupby("segment", as_index=False).size()
    st.plotly_chart(
        px.pie(segment_counts, names="segment", values="size",
               title="Répartition des segments clients"),
        use_container_width=True
    )
    st.dataframe(segment_counts, use_container_width=True)

except Exception as exc:
    st.error("Les tables Gold ne sont pas encore disponibles. Exécute d'abord J1, J2, J3 puis J4.")
    st.exception(exc)
