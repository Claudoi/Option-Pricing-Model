import streamlit as st
from src.models.pricing_binomial import BinomialOption
from src.utils.plot_utils import PlotUtils


def binomial_ui():
    st.markdown("## Binomial Option Pricing")

    with st.form("binomial_form"):
        st.markdown("#### Option Parameters")
        col1, col2 = st.columns(2)
        with col1:
            S = st.number_input("Spot Price (S)", value=100.0, min_value=0.01, format="%.2f")
            K = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, format="%.2f")
            T = st.number_input("Time to Maturity (T in years)", value=1.0, min_value=0.0001, format="%.4f")
            q = st.number_input("Dividend Yield (q)", value=0.0, min_value=0.0, format="%.4f")
        with col2:
            r = st.number_input("Risk-Free Rate (r)", value=0.05, min_value=0.0, format="%.4f")
            sigma = st.number_input("Volatility (σ)", value=0.2, min_value=0.0001, format="%.4f")
            option_type = st.selectbox("Option Type", ["call", "put"])
            N = st.slider("Number of Steps (N)", min_value=1, max_value=100, value=5)
            style = st.selectbox("Option Style", ["European", "American"])

        submitted = st.form_submit_button("Calculate")

    if submitted:
        try:
            bin_opt = BinomialOption(S, K, T, r, sigma, N, option_type, q)
            if style == "European":
                price = bin_opt.price_european()
            else:
                price = bin_opt.price_american()
            st.success(f"Binomial {style} Option Price: {price:.4f}")

            # --- Interactive Plot ---
            st.markdown("#### 📉 Option Price vs Spot")
            fig = PlotUtils.plot_binomial_price_vs_spot(
                K, T, r, sigma, N, option_type, q, BinomialOption
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- Tree Display for small N ---
            if N <= 6:
                st.markdown("#### 🌳 Binomial Tree")
                PlotUtils.show_binomial_tree(S, K, T, r, sigma, N, option_type, q, BinomialOption)

                st.markdown("#### 🧪 Local Sensitivities per Node")
                try:
                    tree = bin_opt.get_sensitivities_tree(american=(style == "American"))
                    dot = PlotUtils.graphviz_binomial_sensitivities(tree)
                    st.graphviz_chart(dot.source)
                except Exception as sensi_err:
                    st.error(f"Error displaying sensitivities: {sensi_err}")

        except Exception as e:
            st.error(f"Error in Binomial pricing: {e}")
