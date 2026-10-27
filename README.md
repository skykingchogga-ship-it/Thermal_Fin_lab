# Fin Thermal Lab

Interactive thermal analysis of extended cooling fins (project #48: *Thermal Analysis of Extended Cooling Fins of Different Shapes and Materials*).

## Run in VS Code

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Validate the physics:

```bash
pytest -v
```

## Files

| File | Purpose |
|---|---|
| `fins.py` | Geometry, analytical solutions (5 shapes), finite-volume solver, metrics, design sweep |
| `app.py` | Streamlit dashboard |
| `viz3d.py` | Rotatable 3D fin models coloured by the 1D temperature profile |
| `solver3d.py` | True 3D steady conduction solver (finite volume, sparse) |
| `viz3d_field.py` | 3D models with cutaway, coloured by the solved 3D field |
| `test_fins.py` | Analytical vs numerical validation and sanity checks |
| `test_3d.py` | 3D solver vs 1D theory, energy balance, behaviour checks |

## Features

- Shapes: rectangular, triangular, parabolic, pin, annular
- Materials: Cu, pure Al, Al 6061, brass, carbon steel, stainless 304, or custom
- Live sliders for geometry, h, base and ambient temperature, tip condition
- Analytical solution validated against a 1D finite-volume solver, with error plot
- Heat dissipation, efficiency, effectiveness, mass, Biot number
- 2D heatmap of the fin temperature field
- Rotatable 3D model of each fin shape coloured by temperature (thickness exaggeration, mounting base toggle)
- True 3D conduction solver: temperature varies along the length and through the thickness/width,
  cutaway view, "Internal ΔT" colouring to expose the hot core, resolution control
- Shape × material comparison (W, efficiency, W/kg)
- Constraint-based optimizer (mass cap, size ranges) with a feasible-design scatter plot

## Method

Solves d/ds(kA dθ/ds) − hPθ = 0 with θ(0) = θ_b and a convective or adiabatic tip.
Closed forms: cosh/sinh (rectangular, pin), Bessel I₀/I₁ (triangular), power law (concave parabolic),
I₀/K₀ (annular). Tapered fins use the thin-fin approximation P = 2w.

**3D solver.** Steady conduction ∇·(k∇T) = 0 with convection h on every exposed face and an isothermal
base. Straight fins: Cartesian finite volumes on one quarter of the fin (symmetry planes adiabatic),
cut cells follow the taper. Pin and annular fins: axisymmetric (r, z) finite volumes, revolved for display.
Validation: matches the 1D solution within 0.03 % for rectangular, pin and annular fins at low Biot number,
closes the energy balance to round-off, and differs by 1–3 % for narrow tapered fins because the 1D thin-fin
form ignores their side faces (the difference vanishes for wide fins).
Tip: try stainless steel, a 10–20 mm thick fin and h = 300 W/m²K to see strong through-thickness gradients.

## Ideas to extend it

- Temperature-dependent k(T) and h(T)
- Fin arrays: spacing and base-area correction
- Export a PDF report of the current design
