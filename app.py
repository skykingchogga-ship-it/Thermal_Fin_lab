"""Fin Thermal Lab — interactive thermal analysis of extended cooling fins.

Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import fins as F
import solver3d
import viz3d
import viz3d_field

st.set_page_config(page_title="Fin Thermal Lab", layout="wide")

STRAIGHT = ("Rectangular", "Triangular", "Parabolic")


@st.cache_data(show_spinner="Solving 3D heat conduction...")
def run_3d(shape, L, t, w, r1, h, k, Tb, Tinf, tip, resolution):
    return solver3d.solve_3d(F.Fin(shape, L, t, w, r1), h, k, Tb, Tinf, tip, resolution)


@st.cache_data(show_spinner="Sweeping design space...")
def run_sweep(shapes, mats, L_range, t_range, h, Tb, Tinf, w, r1, tip, n):
    return F.sweep(list(shapes), list(mats), L_range, t_range, h, Tb, Tinf, w, r1, tip, n)


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Fin Thermal Lab")
    shape = st.selectbox("Fin shape", F.SHAPES)
    mat = st.selectbox("Material", list(F.MATERIALS) + ["Custom"], index=2)
    if mat == "Custom":
        k = st.number_input("Conductivity k (W/m·K)", 1.0, 500.0, 100.0)
        rho = st.number_input("Density (kg/m³)", 500.0, 20000.0, 5000.0)
    else:
        k, rho = F.MATERIALS[mat]["k"], F.MATERIALS[mat]["rho"]
        st.caption(f"k = {k:g} W/m·K  ·  ρ = {rho:g} kg/m³")

    st.subheader("Geometry")
    L = st.slider("Fin height r₂ − r₁ (mm)" if shape == "Annular" else "Fin length (mm)",
                  5.0, 200.0, 50.0, 1.0) / 1e3
    t = st.slider("Base diameter (mm)" if shape == "Pin" else "Base thickness (mm)",
                  0.5, 20.0, 3.0, 0.1) / 1e3
    w = st.slider("Fin width (mm)", 5.0, 200.0, 50.0, 1.0) / 1e3 if shape in STRAIGHT else 0.05
    r1 = st.slider("Base radius r₁ (mm)", 5.0, 80.0, 20.0, 1.0) / 1e3 if shape == "Annular" else 0.02

    st.subheader("Thermal conditions")
    h = st.slider("Convection coefficient h (W/m²·K)", 5.0, 500.0, 25.0, 1.0)
    Tb = st.slider("Base temperature (°C)", 60.0, 500.0, 100.0, 5.0)
    Tinf = st.slider("Ambient temperature (°C)", -10.0, 50.0, 25.0, 1.0)
    tip = st.radio("Tip condition", ["Convective", "Adiabatic"], horizontal=True) == "Convective"
    n = st.select_slider("Finite-volume nodes", [21, 41, 81, 161, 321, 641], value=161)

# --------------------------------------------------------------------------- solve
fin = F.Fin(shape, L, t, w, r1)
g = F.geometry(fin, n)
s = g["s"]
T_an, q_an = F.analytic(fin, h, k, Tb, Tinf, s, tip)
T_num, q_num = F.numeric(fin, h, k, Tb, Tinf, n, tip)
perf = F.performance(fin, h, k, rho, Tb, Tinf, q_an)

st.title("Fin Thermal Lab")
st.caption(f"{shape} fin · {mat} · analytical solution validated against a 1D finite-volume solver")

c = st.columns(6)
c[0].metric("Heat dissipation", f"{q_an:.2f} W")
c[1].metric("Efficiency η", f"{perf['eta'] * 100:.1f} %")
c[2].metric("Effectiveness ε", f"{perf['eps']:.1f}")
c[3].metric("Tip temperature", f"{T_an[-1]:.1f} °C")
c[4].metric("Fin mass", f"{perf['mass'] * 1e3:.1f} g")
c[5].metric("Biot number", f"{perf['biot']:.3f}",
            help="Bi = h(t/2)/k. The 1D fin model is reliable for Bi < 0.1.")
if perf["biot"] > 0.1:
    st.warning("Biot number > 0.1: temperature varies noticeably across the thickness, "
               "so the 1D fin assumption is weak for this design.")

tab_prof, tab_map, tab_3d, tab_fld, tab_cmp, tab_opt, tab_theory = st.tabs(
    ["Temperature & validation", "Heatmap", "3D model", "3D field solver",
     "Compare shapes × materials", "Optimizer", "Theory"])

# --------------------------------------------------------------------------- tab 1
with tab_prof:
    step = max(1, n // 40)
    left, right = st.columns(2)

    fig = go.Figure()
    fig.add_scatter(x=s * 1e3, y=T_an, name="Analytical", line=dict(width=3, color="#e4572e"))
    fig.add_scatter(x=s[::step] * 1e3, y=T_num[::step], name="Finite volume", mode="markers",
                    marker=dict(size=8, symbol="circle-open", color="#17bebb", line=dict(width=2)))
    fig.update_layout(title="Temperature profile", xaxis_title="Distance from base (mm)",
                      yaxis_title="Temperature (°C)", height=380,
                      legend=dict(orientation="h", y=1.12), margin=dict(t=70))
    left.plotly_chart(fig, width="stretch")

    Q = -k * g["A"] * np.gradient(T_num, s)
    fig = go.Figure(go.Scatter(x=s * 1e3, y=Q, fill="tozeroy", line=dict(color="#f2a541", width=3)))
    fig.update_layout(title="Conducted heat flow along the fin", xaxis_title="Distance from base (mm)",
                      yaxis_title="Q(s) (W)", height=380, margin=dict(t=70))
    right.plotly_chart(fig, width="stretch")

    err = np.abs(T_num - T_an)
    fig = go.Figure(go.Scatter(x=s * 1e3, y=err, fill="tozeroy", line=dict(color="#6a4c93")))
    fig.update_layout(title="Validation: |T_numerical − T_analytical|", xaxis_title="Distance from base (mm)",
                      yaxis_title="Error (K)", height=280, margin=dict(t=60))
    st.plotly_chart(fig, width="stretch")

    v = st.columns(3)
    v[0].metric("Max temperature error", f"{err.max():.4f} K")
    v[1].metric("q analytical / numerical", f"{q_an:.3f} / {q_num:.3f} W")
    v[2].metric("Heat-flow difference", f"{abs(q_num - q_an) / q_an * 100:.3f} %")
    st.caption("Increase the number of nodes in the sidebar and watch the error shrink.")

# --------------------------------------------------------------------------- tab 2
with tab_map:
    if shape == "Annular":
        r2 = r1 + L
        xs = np.linspace(-r2, r2, 500)
        X, Y = np.meshgrid(xs, xs)
        R = np.hypot(X, Y)
        Z = np.interp(R, r1 + s, T_an)
        Z[(R < r1) | (R > r2)] = np.nan
        fig = go.Figure(go.Heatmap(x=xs * 1e3, y=xs * 1e3, z=Z, colorscale="Inferno",
                                   colorbar=dict(title="°C"), zmin=Tinf, zmax=Tb))
        fig.update_yaxes(scaleanchor="x", title="y (mm)")
        fig.update_xaxes(title="x (mm)")
        fig.update_layout(title="Annular fin — top view", height=620)
        note = "Top view of the ring; temperature depends on radius only."
    else:
        xs = np.linspace(0, L, 500)
        tmax = g["th"].max()
        ys = np.linspace(-tmax / 2, tmax / 2, 160)
        half = np.interp(xs, s, g["th"]) / 2
        Z = np.tile(np.interp(xs, s, T_an), (len(ys), 1))
        Z[np.abs(ys)[:, None] > half[None, :]] = np.nan
        fig = go.Figure(go.Heatmap(x=xs * 1e3, y=ys * 1e3, z=Z, colorscale="Inferno",
                                   colorbar=dict(title="°C"), zmin=Tinf, zmax=Tb))
        fig.update_xaxes(title="Distance from base (mm)")
        fig.update_yaxes(title="Thickness (mm, exaggerated)")
        fig.update_layout(title=f"{shape} fin — side view", height=420)
        note = "Side view with the thickness exaggerated for visibility."
    st.plotly_chart(fig, width="stretch")
    st.caption(f"{note} The colours map the 1D solution onto the fin outline "
               "(valid while the Biot number is small).")

# --------------------------------------------------------------------------- tab 3d
with tab_3d:
    o1, o2, o3 = st.columns(3)
    src = o1.radio("Temperature source", ["Analytical", "Finite volume"], horizontal=True)
    ex = o2.slider("Thickness exaggeration ×", 1.0, 10.0, 1.0, 0.5,
                   help="Scales the thickness (or pin radius) so thin fins are easier to see.")
    wall = o3.checkbox("Show mounting base", True)
    fig3d = viz3d.fin_3d_figure(fin, s, T_an if src == "Analytical" else T_num, Tinf, Tb, ex=ex, wall=wall)
    st.plotly_chart(fig3d, width="stretch")
    st.caption("Drag to rotate, scroll to zoom, hover to read the local temperature. "
               "Temperature varies along the fin only (1D model); the grey part is the mounting base, "
               "not part of the calculation.")

# --------------------------------------------------------------------------- tab 3d field
with tab_fld:
    st.markdown("Full **3D steady conduction** solved through the volume of the fin "
                "(not the 1D approximation): temperature varies along the length *and* across the "
                "thickness/width. Heat leaves every exposed surface by convection.")
    f1, f2, f3, f4, f5 = st.columns([1, 1.4, 1, 1.4, 1])
    resolution = f1.selectbox("Resolution", ["Coarse", "Medium", "Fine"], index=1)
    mode_lbl = f2.radio("Colour by", ["Temperature", "Internal ΔT (hot core)"], horizontal=True,
                        help="Internal ΔT = temperature minus the coolest point of the same cross-section. "
                             "It reveals the 3D structure even when the through-thickness variation is small.")
    cut = f3.checkbox("Cutaway", True)
    ex_f = f4.slider("Thickness exaggeration ×", 1.0, 10.0, 1.0, 0.5, key="ex_field")
    wall_f = f5.checkbox("Show base", True, key="wall_field")

    r3 = run_3d(shape, L, t, w, r1, h, k, Tb, Tinf, tip, resolution)
    q1d = q_an
    m = st.columns(5)
    m[0].metric("Heat flow, 3D solver", f"{r3['q']:.3f} W")
    m[1].metric("Heat flow, 1D model", f"{q1d:.3f} W", delta=f"{(r3['q'] / q1d - 1) * 100:+.2f} % (3D vs 1D)",
                delta_color="off")
    m[2].metric("Tip temperature, 3D", f"{r3['T_tip']:.2f} °C")
    m[3].metric("Max internal ΔT in a section", f"{r3['dT_cross']:.4f} K")
    m[4].metric("Energy balance error", f"{abs(r3['q'] - r3['q_conv']) / r3['q'] * 100:.1e} %",
                help="Heat in at the base vs heat out through all convecting surfaces.")

    fig = viz3d_field.field_figure(fin, r3, Tinf, Tb, "T" if mode_lbl == "Temperature" else "dev",
                                   ex=ex_f, wall=wall_f, cut=cut)
    st.plotly_chart(fig, width="stretch")
    if r3["dT_cross"] < 0.5:
        st.info(f"At Biot = {perf['biot']:.3f} the temperature across each section differs by only "
                f"{r3['dT_cross']:.3f} K, so the temperature view looks 1D. Switch to **Internal ΔT** to see the "
                "3D structure, or try stainless steel, a thicker fin and a larger h to see strong 3D effects.")
    st.caption("Straight fins: one quarter is solved (symmetry planes) with cut cells following the taper. "
               "Pin and annular fins: exact axisymmetric solution revolved for display. "
               "Tapered fins carry ~1–3 % more heat than the 1D model because the 1D thin-fin form "
               "ignores the side faces.")

# --------------------------------------------------------------------------- tab 3
with tab_cmp:
    rows = []
    for sh in F.SHAPES:
        f = F.Fin(sh, L, t, w, r1)
        for mname, p in F.MATERIALS.items():
            _, q = F.analytic(f, h, p["k"], Tb, Tinf, np.array([0.0, L]), tip)
            pf = F.performance(f, h, p["k"], p["rho"], Tb, Tinf, q)
            rows.append({"Shape": sh, "Material": mname, "q (W)": q, "η (%)": pf["eta"] * 100,
                         "ε": pf["eps"], "Mass (g)": pf["mass"] * 1e3, "q/mass (W/kg)": q / pf["mass"]})
    df = pd.DataFrame(rows)

    a, b = st.columns(2)
    fig = px.bar(df, x="Shape", y="q (W)", color="Material", barmode="group",
                 title="Heat dissipation", category_orders={"Shape": F.SHAPES})
    fig.update_layout(height=400, legend=dict(orientation="h", y=-0.2))
    a.plotly_chart(fig, width="stretch")

    piv = df.pivot(index="Material", columns="Shape", values="η (%)").reindex(columns=F.SHAPES)
    fig = px.imshow(piv, text_auto=".1f", aspect="auto", color_continuous_scale="Viridis",
                    title="Fin efficiency η (%)")
    fig.update_layout(height=400)
    b.plotly_chart(fig, width="stretch")

    fig = px.bar(df, x="Shape", y="q/mass (W/kg)", color="Material", barmode="group",
                 title="Heat dissipated per kg of fin", category_orders={"Shape": F.SHAPES})
    fig.update_layout(height=380, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

    st.dataframe(df, width="stretch", hide_index=True, column_config={
        "q (W)": st.column_config.NumberColumn(format="%.2f"),
        "η (%)": st.column_config.NumberColumn(format="%.1f"),
        "ε": st.column_config.NumberColumn(format="%.1f"),
        "Mass (g)": st.column_config.NumberColumn(format="%.1f"),
        "q/mass (W/kg)": st.column_config.NumberColumn(format="%.0f"),
    })
    st.caption("All shapes share the sidebar's length, base thickness/diameter, width and r₁, "
               "so tapered fins weigh less: compare W/kg as well as W. Straight fins are per "
               "the chosen width; pin and annular fins are per fin.")

# --------------------------------------------------------------------------- tab 4
with tab_opt:
    st.subheader("Best fin for your constraints")
    oc, orr = st.columns([1, 2])
    with oc:
        objective = st.radio("Objective", ["Maximize heat dissipation", "Maximize heat per unit mass"])
        mass_cap = st.number_input("Max fin mass (g)", 1.0, 5000.0, 100.0, 5.0)
        L_rng = st.slider("Length range (mm)", 5.0, 200.0, (20.0, 100.0))
        t_rng = st.slider("Base thickness / diameter range (mm)", 0.5, 20.0, (1.0, 8.0), 0.1)
        shapes_sel = st.multiselect("Shapes", F.SHAPES, default=F.SHAPES)
        mats_sel = st.multiselect("Materials", list(F.MATERIALS), default=list(F.MATERIALS))
        need_bi = st.checkbox("Require Biot ≤ 0.1 (1D model valid)", True)
        st.caption("Width, r₁, h and temperatures come from the sidebar.")

    with orr:
        if not shapes_sel or not mats_sel:
            st.info("Select at least one shape and one material.")
        else:
            res = run_sweep(tuple(shapes_sel), tuple(mats_sel),
                            (L_rng[0] / 1e3, L_rng[1] / 1e3), (t_rng[0] / 1e3, t_rng[1] / 1e3),
                            h, Tb, Tinf, w, r1, tip, 14)
            ok = res[res["Mass (g)"] <= mass_cap]
            if need_bi:
                ok = ok[ok["Biot"] <= 0.1]
            if ok.empty:
                st.error("No design satisfies the constraints. Raise the mass cap or widen the ranges.")
            else:
                key = "q (W)" if objective.startswith("Maximize heat dissipation") else "q/mass (W/kg)"
                ok = ok.sort_values(key, ascending=False)
                best = ok.iloc[0]
                m = st.columns(4)
                m[0].metric("Best shape", best["Shape"])
                m[1].metric("Material", best["Material"])
                m[2].metric("L × t", f"{best['L (mm)']:.0f} × {best['t (mm)']:.1f} mm")
                m[3].metric("Heat", f"{best['q (W)']:.2f} W")
                m = st.columns(3)
                m[0].metric("Efficiency", f"{best['η (%)']:.1f} %")
                m[1].metric("Mass", f"{best['Mass (g)']:.1f} g")
                m[2].metric("Heat per kg", f"{best['q/mass (W/kg)']:.0f} W/kg")

                fig = px.scatter(ok, x="Mass (g)", y="q (W)", color="Shape", symbol="Material",
                                 opacity=0.55, hover_data=["L (mm)", "t (mm)", "η (%)"],
                                 title=f"{len(ok)} feasible designs")
                fig.add_scatter(x=[best["Mass (g)"]], y=[best["q (W)"]], mode="markers", name="Optimum",
                                marker=dict(symbol="star", size=20, color="gold",
                                            line=dict(width=2, color="black")))
                fig.update_layout(height=430)
                st.plotly_chart(fig, width="stretch")

    if shapes_sel and mats_sel and "ok" in locals() and not ok.empty:
        st.markdown("**Top 10 designs**")
        st.dataframe(ok.head(10).drop(columns="Biot"), width="stretch", hide_index=True,
                     column_config={c: st.column_config.NumberColumn(format="%.2f") for c in
                                    ["L (mm)", "t (mm)", "q (W)", "η (%)", "Mass (g)", "q/mass (W/kg)"]})

# --------------------------------------------------------------------------- tab 5
with tab_theory:
    st.markdown("**Governing equation** (steady 1D conduction with lateral convection), "
                "θ = T − T∞, s = distance from the base:")
    st.latex(r"\frac{d}{ds}\left(kA(s)\frac{d\theta}{ds}\right) - hP(s)\,\theta = 0,\qquad "
             r"\theta(0)=\theta_b,\quad -k\theta'(L)=h\,\theta(L)")
    st.markdown("""
**Closed-form solutions used**

| Shape | Solution |
|---|---|
| Rectangular / Pin | cosh/sinh with convective tip, m = √(hP/kA) |
| Triangular | Modified Bessel I₀, I₁ with m = √(2h/kt) |
| Parabolic (concave, t ∝ x²) | Power law θ ∝ xʳ, r = (−1+√(1+4(mL)²))/2 |
| Annular | I₀/K₀ combination with base and tip boundary conditions |

**Metrics**
- Efficiency: η = q / (h·A_surf·θ_b)
- Effectiveness: ε = q / (h·A_base·θ_b)
- Biot number: Bi = h(t/2)/k, so the 1D model needs Bi ≲ 0.1

**Numerical check:** a finite-volume solver on a uniform grid with variable A(s) and P(s)
reproduces every analytical profile, and the error plot shows convergence as nodes increase.

**Assumptions:** steady state, constant k and h, negligible radiation, uniform ambient, no contact
resistance at the base. Tapered fins use the thin-fin surface approximation P = 2w.
""")
