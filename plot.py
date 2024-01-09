from scipy.ndimage import gaussian_filter
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# use latex
plt.rc("text", usetex=True)
plt.rc("font", family="Palatino")

data = {}
with open("data/raw.csv", "r") as f:
    columns = f.readline().strip().split(",")
    data = {k: [] for k in columns}
    for line in f:
        for k, v in zip(columns, line.strip().split(",")):
            if v.replace(".", "", 1).replace("e", "", 1).replace("-", "", 1).isdigit():
                data[k].append(float(v))
            elif v == "":
                data[k].append(None)
            else:
                data[k].append(v)

for k in data:
    data[k] = np.array(data[k])
    if k in [
        "GLON",
        "GLAT",
        "P0",
        "dP0",
        "P1",
        "dP1",
        "Dist",
        "Edot",
        "Bsurf",
        "BLC",
        "RLC",
        "gmax",
    ]:
        data[k] = data[k].astype(float)


# available columns:
#   Source_Name
#   RAJ2000
#   DEJ2000
#   GLON -- galactic longitude
#   GLAT -- galactic latitude
#   P0 -- period [s]
#   dP0 -- period error
#   P1 -- spindown
#   dP1 -- spindown error
#   Dist -- distance [kpc]
#   Edot -- spindown luminosity [erg/s]
#   Bsurf -- surface magnetic field [G]
#   Catalog -- "Fermi" or "ATNF"
#   BLC -- magnetic field at light cylinder [G]
#   RLC -- light cylinder radius [cm]
#   gmax -- polar cap voltage [m_e c^2]
#   gradLC -- gamma_rad at LC

# pulsars with special properties
intermittents = dict()
intermittents["p"], intermittents["pdot"] = np.loadtxt(
    "data/intermittent.dat", usecols=(1, 2), unpack=True
)

rrats = dict()
rrats["p"], rrats["pdot"] = np.loadtxt("data/rrats.dat", usecols=(1, 2), unpack=True)

nullings = dict()
nullings["p"], nullings["pdot"] = np.loadtxt(
    "data/nulling.dat", usecols=(1, 2), unpack=True
)

giants = dict(
    p=np.array([0.002323, 0.033392, 0.050570, 0.005440, 0.003054, 0.001558, 0.001607])
    * 1e3,
    pdot=[7.74e-20, 4.21e-13, 4.79e-13, 3.38e-18, 1.62e-18, 1.05e-19, 1.69e-20],
)

# plotting properties

props = dict(
    xlim=(5e-4, 20),
    ylim=(1e-21, 1e-10),
    xscale="log",
    yscale="log",
    xlabel="Period [s]",
    ylabel=r"$\dot{P}$ [s/s]",
)


radio_props = dict(c="darkgrey", s=2, lw=0.2, edgecolor="k", zorder=9)
fermi_props = dict(s=10, marker="D", facecolor="C1", edgecolors="k", lw=0.5, zorder=10)
giant_props = dict(
    s=45,
    marker="X",
    facecolor="C1",
    edgecolors="k",
    lw=0.5,
    zorder=11,
)
null_props = dict(
    s=20,
    marker="o",
    facecolor="C0",
    edgecolors="k",
    lw=0.5,
    zorder=13,
)
rrat_props = dict(
    s=20,
    marker="o",
    facecolor="C4",
    edgecolors="k",
    lw=0.5,
    zorder=14,
)
int_props = dict(
    s=20,
    marker="o",
    facecolor="C3",
    edgecolors="k",
    lw=0.5,
    zorder=15,
)

# plotting
fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
# ax.grid(True, c="k")

radio_mask = data["Catalog"] == "ATNF"
fermi_mask = data["Catalog"] == "Fermi"

ax.scatter(data["P0"][fermi_mask], data["P1"][fermi_mask], **fermi_props)
ax.scatter(data["P0"][radio_mask], data["P1"][radio_mask], **radio_props)
ax.scatter(giants["p"] / 1e3, giants["pdot"], **giant_props)
ax.scatter(nullings["p"], nullings["pdot"], **null_props)
ax.scatter(rrats["p"], rrats["pdot"] * 1e-15, **rrat_props)
ax.scatter(intermittents["p"], intermittents["pdot"], **int_props)
ax.set(**props)

# annotations
# # 1D gap line
xs = np.logspace(2, 5, 100) / 1e3
psi = 0.65
ys = 4e-15 * (xs) ** (11 / 4) * psi ** (-7 / 2)
prop = dict(c="C3", alpha=0.25)
ax.plot(xs, ys, lw=30, zorder=-1, **prop)
ax.plot([], [], label=r"$1$D gap approximation", lw=5, **prop)

# # 1D gap arrow
ind = 1
i1, j1 = ax.transData.transform((xs[ind], ys[ind]))
i2, j2 = ax.transData.transform((xs[ind + 1], ys[ind + 1]))
ai, aj = (i2 - i1, j2 - j1)
x2, y2 = ax.transData.inverted().transform((i1 - 10 * aj, j1 + 10 * ai))
x1, y1 = ax.transData.inverted().transform((i1 - 4.5 * aj, j1 + 4.5 * ai))
ax.annotate(
    "",
    xy=(x2, y2),
    xycoords="data",
    xytext=(x1, y1),
    textcoords="data",
    arrowprops=dict(color="C3", alpha=0.75, lw=0),
)

# # LC pp line
xs = np.logspace(np.log10(20), 5, 100) / 1e3
ys = 5e-11 * (xs) ** (7 / 2)
ax.plot(xs, ys, c="C0", zorder=-1, lw=5, alpha=0.5, label=r"LC pair production")

# # LC pp arrow
ind = 5
i1, j1 = ax.transData.transform((xs[ind], ys[ind]))
i2, j2 = ax.transData.transform((xs[ind + 1], ys[ind + 1]))
ai, aj = (i2 - i1, j2 - j1)
x2, y2 = ax.transData.inverted().transform((i1 - 1.2 * 5 * aj, j1 + 1.5 * 3 * ai))
x1, y1 = ax.transData.inverted().transform((i1 - 5 * 0.18 * aj, j1 + 3 * 0.18 * ai))
ax.annotate(
    "",
    xy=(x2, y2),
    xycoords="data",
    xytext=(x1, y1),
    textcoords="data",
    arrowprops=dict(color="C0", alpha=0.5, lw=0),
)

# # Edot lines
xs = np.logspace(-1, 5, 100) / 1e3
ys = np.logspace(-22, -9, 100)
x, y = np.meshgrid(xs, ys)
z = 4 * np.pi**2 * 1e45 * y / x**3
cs = ax.contour(
    x,
    y,
    z,
    levels=np.logspace(30, 38, 3),
    colors="C1",
    linewidths=1,
    zorder=0,
    alpha=0.75,
)
fmt = mpl.ticker.LogFormatterMathtext()
fmt.create_dummy_axis()
manual_locations = [(100 / 1e3, 1e-10), (2500 / 1e3, 1e-10), (1e4 / 1e3, 1e-13)]
ax.clabel(
    cs,
    cs.levels,
    inline=True,
    fmt=fmt,
    manual=manual_locations,
    inline_spacing=10,
    fontsize=8,
)
ax.plot([], [], c="C1", lw=1, alpha=0.75, label=r"$\dot{E}$ [erg/s]")

# # Bsurf lines
xs = np.logspace(-1, 5, 100) / 1e3
ys = np.logspace(-22, -9, 100)
x, y = np.meshgrid(xs, ys)
z = 2 * np.sqrt(1e45 * (3e10) ** 3 * (x * y)) / (2 * np.pi * (1.2e6) ** 3)
cs = ax.contour(
    x, y, z, levels=[1e11, 1e13], colors="C2", linewidths=1, zorder=0, alpha=0.75
)
fmt = mpl.ticker.LogFormatterMathtext()
fmt.create_dummy_axis()
manual_locations = [(1.2e4 / 1e3, 2e-18), (1.2e4 / 1e3, 1e-14)]
ax.clabel(
    cs,
    cs.levels,
    inline=True,
    fmt=fmt,
    manual=manual_locations,
    inline_spacing=10,
    fontsize=8,
)
ax.plot([], [], c="C2", lw=1, alpha=0.75, label=r"$B_*$ [G]")

# # BLC lines
x, y = np.meshgrid(xs, ys)
z = 2 * np.sqrt(1e45 * (3e10) ** 3 * (x * y)) / (2 * np.pi * (1.2e6) ** 3)
z = z * (1e6 / (3e10 / (2 * np.pi * (1 / x)))) ** 3
cs = ax.contour(
    x,
    y,
    z,
    levels=[1e5],
    linestyles="--",
    colors="C2",
    linewidths=1,
    zorder=0,
    alpha=0.75,
)
fmt = mpl.ticker.LogFormatterMathtext()
fmt.create_dummy_axis()
manual_locations = [(0.01, 2e-11)]
ax.clabel(
    cs,
    cs.levels,
    inline=True,
    fmt=fmt,
    manual=manual_locations,
    inline_spacing=10,
    fontsize=8,
    use_clabeltext=True,
)
ax.plot([], [], c="C2", lw=1, ls="--", alpha=0.75, label=r"$B_{\rm LC}$ [G]")


# # background
xedges = np.logspace(-1, 5, 200) / 1e3
yedges = np.logspace(-23, -8, 200)
H, _, _ = np.histogram2d(data["P0"], data["P1"], bins=(xedges, yedges))
xcenters = (xedges[:-1] + xedges[1:]) / 2
ycenters = (yedges[:-1] + yedges[1:]) / 2
ax.contourf(
    xcenters,
    ycenters,
    gaussian_filter(H.T, 3),
    levels=np.linspace(0, 2.5, 10),
    cmap="Greys",
    zorder=-2,
)


# legends
ax.legend(loc="upper left")

ax2 = ax.twinx()
ax2.get_yaxis().set_visible(False)
ax2.scatter([], [], **fermi_props, label=r"$\gamma$-ray pulsars")
ax2.scatter([], [], **giant_props, label=r"pulsars with giant pulses")
ax2.scatter([], [], **int_props, label=r"intermittent pulsars")
ax2.scatter([], [], **rrat_props, label=r"RRATs")
ax2.scatter([], [], **null_props, label=r"pulsars with nullings")
ax2.scatter([], [], **radio_props, label=r"all radiopulsars \& magnetars")
# set frame without border but with background
lg = ax2.legend(loc="lower right", fontsize=8, frameon=False)
plt.savefig("ppdot.png", bbox_inches="tight")
plt.tight_layout()
# plt.show()
