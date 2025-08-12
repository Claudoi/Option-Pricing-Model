import streamlit as st
from src.models.pricing_binomial import BinomialOption
from src.utils.plot_utils import PlotUtils

# --- Cache the model computation to avoid recomputing on every rerun ---
@st.cache_data(show_spinner=False)
def _compute_node_tree_and_price(S, K, T, r, sigma, N, option_type, q, american: bool):
    """
    Compute the option price and sensitivity tree for given parameters.
    Caches results to improve responsiveness when only sliders change.
    """
    model = BinomialOption(S, K, T, r, sigma, N, option_type, q)
    price = model.price_european() if not american else model.price_american()
    node_tree = model.get_sensitivities_tree(american=american)
    return price, node_tree

def binomial_ui():
    st.markdown("## Binomial Option Pricing")

    # ---------- INPUT FORM ----------
    # Collects parameters and triggers computation when "Calculate" is pressed
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
            N = st.slider("Number of Steps (N)", min_value=1, max_value=200, value=10)
            style = st.selectbox("Option Style", ["European", "American"])

        # Visualization toggles (persisted in session_state)
        st.checkbox("Show binomial tree", value=True, key="show_tree")
        st.checkbox("Show node table", value=False, key="show_table")
        st.checkbox("Show axes/grid", value=True, key="show_axes")
        st.checkbox("Show level labels", value=True, key="show_level_labels")

        submitted = st.form_submit_button("Calculate")

    # ---------- COMPUTE & STORE IN SESSION ----------
    # Only computes when "Calculate" is pressed, keeps results in session_state
    if submitted:
        try:
            american = (style == "American")
            price, node_tree = _compute_node_tree_and_price(S, K, T, r, sigma, N, option_type, q, american)
            st.session_state.binomial = {
                "params": dict(S=S, K=K, T=T, r=r, sigma=sigma, N=N,
                               option_type=option_type, q=q, american=american),
                "price": float(price),
                "node_tree": node_tree
            }
            # Initialize slider positions for node selection
            st.session_state.level_i = min(st.session_state.get("level_i", N // 2), N)
            st.session_state.index_j = min(st.session_state.get("index_j", st.session_state.level_i // 2),
                                           st.session_state.level_i)
        except Exception as e:
            st.error(f"Error in Binomial pricing: {e}")

    # ---------- RENDER SECTION ----------
    # Uses stored results in session_state so graphs don't disappear on slider change
    data = st.session_state.get("binomial")
    if not data:
        st.info("Set parameters and click **Calculate** to compute the tree.")
        return

    # Display computed price
    st.success(f"Binomial {'American' if data['params']['american'] else 'European'} Option Price: {data['price']:.4f}")

    # --- Price vs Spot Chart ---
    st.markdown("#### Option Price vs Spot")
    p = data["params"]
    fig_vs_spot = PlotUtils.plot_binomial_price_vs_spot(
        p["K"], p["T"], p["r"], p["sigma"], p["N"], p["option_type"], p["q"], BinomialOption
    )
    st.plotly_chart(fig_vs_spot, use_container_width=True)

    # --- Binomial Tree with interactive node selection ---
    if st.session_state.show_tree and data["params"]["N"] <= 120:
        st.markdown("#### Binomial Tree")

        node_tree = data["node_tree"]
        N = data["params"]["N"]

        # Independent sliders for node location (persist values between reruns)
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            level_i = st.slider("Level (i)", 0, N,
                                value=st.session_state.get("level_i", min(N // 2, N)),
                                key="level_i")
        with c2:
            # Ensure j is clamped to current i
            max_j = level_i
            default_j = min(st.session_state.get("index_j", level_i // 2), max_j)
            index_j = st.slider("Index (j)", 0, max_j, value=default_j, key="index_j")
        with c3:
            st.caption("Use the sliders to locate a node. The figure highlights the node, its level and the path from the root.")

        # Clamp index_j if i changes
        if st.session_state.index_j > level_i:
            st.session_state.index_j = level_i
            index_j = level_i

        # Plot tree with selected node highlighted
        fig_tree = PlotUtils.plot_binomial_tree_from_nodes(
            node_tree,
            title="Binomial Tree",
            show_axes=st.session_state.show_axes,
            show_grid=st.session_state.show_axes,
            show_level_labels=st.session_state.show_level_labels,
            highlight_node=(level_i, index_j),
            highlight_path=True,
            show_indices=True
        )
        st.plotly_chart(fig_tree, use_container_width=True)

        # --- Node Info Metrics ---
        node = node_tree[level_i][index_j]
        st.metric(label="Selected Node", value=f"(i={level_i}, j={index_j})")
        cA, cB, cC, cD = st.columns(4)
        cA.metric("S", f"{node['S']:.4f}")
        cB.metric("V", f"{node['V']:.4f}")
        cC.metric("Δ", f"{node['Delta']:.6f}" if node['Delta'] == node['Delta'] else "—")
        cD.metric("Γ", f"{node['Gamma']:.6f}" if node['Gamma'] == node['Gamma'] else "—")

    elif st.session_state.show_tree and data["params"]["N"] > 120:
        st.info("Tree hidden for large N to keep the app responsive. Reduce N to display the tree.")

    # --- Optional Table View ---
    if st.session_state.show_table:
        rows = []
        for i, level in enumerate(data["node_tree"]):
            for j, node in enumerate(level):
                rows.append({"level": i, "index": j, **node})
        st.dataframe(rows, use_container_width=True)