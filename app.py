import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from NN_relu import SimpleNeuralNetwork

st.set_page_config(page_title="Neural Network for Soil Mosture", layout="wide")


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


def main():
    st.title("Neural Network for Soil Mosture")
    st.caption("Regresi Soil Moisture menggunakan NN ReLU")

    st.sidebar.header("Random Seed")
    SEED = st.sidebar.number_input("Random Seed", value=42, step=1)
    np.random.seed(SEED)

    df = load_data("synthetic_soil_moisture_5000.csv")

    st.subheader("Preview Dataset")
    st.dataframe(df.head())

    target = "SoilMoisture"
    feature_cols = [c for c in df.columns if c != target]

    X = df[feature_cols].values
    y = df[[target]].values

    st.sidebar.header("Training Config")
    hidden_size = st.sidebar.slider("Hidden Layer Size", 1, 64, 8)
    epochs = st.sidebar.slider("Epochs", 500, 10000, 3000, step=500)
    lr = st.sidebar.number_input("Learning Rate", value=0.0001, format="%.6f")
    test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)

    n = len(X)
    idx = np.random.permutation(n)
    split = int(n * (1 - test_size))

    train_idx, test_idx = idx[:split], idx[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    if st.button("Latih Model"):
        with st.spinner("Training sedang berjalan..."):
            model = SimpleNeuralNetwork(
                input_size=X_train.shape[1],
                hidden_size=hidden_size,
                output_size=1
            )
            model.train(X_train, y_train, epochs=epochs, lr=lr)

        st.session_state.model = model
        st.session_state.scaler_X = scaler_X
        st.session_state.scaler_y = scaler_y

        y_pred_test = model.forward(X_test)

        mse = mean_squared_error(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)

        c1, c2, c3 = st.columns(3)
        c1.metric("MSE", f"{mse:.5f}")
        c2.metric("MAE", f"{mae:.5f}")
        c3.metric("R²", f"{r2:.5f}")

    st.sidebar.header("Metrik & Visualisasi")
    show_mse = st.sidebar.checkbox("Tampilkan MSE", value=True)
    show_mae = st.sidebar.checkbox("Tampilkan MAE", value=True)
    show_r2 = st.sidebar.checkbox("Tampilkan R²", value=True)

    if "model" in st.session_state:
        model = st.session_state.model

        data_plot = {}
        if show_mse:
            data_plot["MSE"] = model.loss_history
        if show_mae:
            data_plot["MAE"] = model.mae_history
        if show_r2:
            data_plot["R²"] = model.r2_history

        df_plot = pd.DataFrame(data_plot)
        df_plot.index += 1
        df_plot.index.name = "Epoch"

        max_epoch_view = st.slider(
            "Tampilkan sampai epoch:",
            min_value=1,
            max_value=len(df_plot),
            value=len(df_plot),
        )

        st.subheader("Training Curve (Interaktif)")
        st.line_chart(df_plot.iloc[:max_epoch_view])

    st.subheader("Prediksi Data Baru")

    if "model" in st.session_state:
        inputs = []
        cols = st.columns(3)

        for i, col in enumerate(feature_cols):
            with cols[i % 3]:
                val = st.number_input(col, float(df[col].median()))
                inputs.append(val)

        if st.button("Prediksi"):
            x_new = np.array(inputs).reshape(1, -1)
            x_new = st.session_state.scaler_X.transform(x_new)
            y_scaled = st.session_state.model.forward(x_new)
            y_pred = st.session_state.scaler_y.inverse_transform(y_scaled)
            st.success(f"Soil Moisture Prediksi: {y_pred[0][0]:.2f}")
    else:
        st.info("Latih model terlebih dahulu")


if __name__ == "__main__":
    main()
