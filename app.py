import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Phân loại khách hàng", layout="wide")
st.title("🛍️ Ứng dụng phân loại khách hàng tiềm năng")

with st.sidebar:
    st.header("📁 Tải dữ liệu & cấu hình")
    uploaded_file = st.file_uploader("Tải lên file CSV", type="csv")
    selected_features = st.multiselect(
        "Chọn thuộc tính để phân cụm:",
        ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'],
        default=['Annual Income (k$)', 'Spending Score (1-100)']
    )
    k = st.slider("Chọn số cụm (k)", min_value=2, max_value=10, value=5)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Lỗi khi đọc file: {e}")
        st.stop()

    st.subheader("📋 Dữ liệu đầu vào")
    st.dataframe(df.head())

    if len(selected_features) < 2:
        st.warning("⚠️ Vui lòng chọn ít nhất 2 thuộc tính để phân cụm.")
        st.stop()

    X = df[selected_features]

    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(X)

    st.subheader("📊 Biểu đồ phân cụm khách hàng")
    if len(selected_features) == 2:
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=selected_features[0], y=selected_features[1], hue='Cluster', palette='Set2', ax=ax)
        ax.set_title("Biểu đồ Scatter phân cụm")
        st.pyplot(fig)
    else:
        st.write("🔍 Biểu đồ đa chiều (Pairplot):")
        sns_plot = sns.pairplot(df, hue='Cluster', vars=selected_features, palette='Set2')
        st.pyplot(sns_plot)

    st.subheader("📦 Phân bố khách hàng theo từng cụm")
    cluster_counts = df['Cluster'].value_counts().sort_index()
    fig2, ax2 = plt.subplots()
    ax2.pie(cluster_counts, labels=[f"Cụm {i}" for i in cluster_counts.index], autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')
    st.pyplot(fig2)

    st.subheader("📈 Trung bình các thuộc tính theo từng cụm")
    st.dataframe(df.groupby('Cluster')[selected_features].mean(numeric_only=True).round(2))

    st.subheader("📥 Tải xuống kết quả phân cụm")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Tải file CSV kết quả",
        data=csv,
        file_name='phan_cum_khach_hang.csv',
        mime='text/csv'
    )

    st.success("✅ Phân loại khách hàng hoàn tất!")

else:
    st.info("👈 Vui lòng tải lên file CSV ở thanh bên để bắt đầu.")
