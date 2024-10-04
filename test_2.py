# %%
from src.solver import AssetOptionSolver, BarrierAssetOptionSolver, ForexOptionSolver

S = {0: [12.0], 1: [9.0, 15.0], 2: [6.75, 11.25, 18.75]}
R = 1.05
pi = 0.6
T = 2
K = 10.6


# %% Question 2
american_model = AssetOptionSolver(expire=T, S=S, R=R, pi=pi, K=K, type="put", style="american")
print(f"Question 2 premium: {american_model.derivative.premium}")

# %% Question 3
B = 10.7
K = 10.7
T = 2
R = 1.05
pi = 0.6
down_in_model = BarrierAssetOptionSolver(
    expire=T,
    S=S,
    R=R,
    pi=pi,
    K=K,
    type="call",
    style="european",
    B=B,
    strike=K,
    barrier_type="down and in",
)
print(f"Question 3 premium: {down_in_model.derivative.premium}")
# %% Question 4
B = 7.2
K = 9.5
T = 2
R = 1.05
pi = 0.6
down_out_model = BarrierAssetOptionSolver(
    expire=T,
    S=S,
    R=R,
    pi=pi,
    K=K,
    type="put",
    style="american",
    B=B,
    strike=K,
    barrier_type="down and out",
)
print(f"Question 4 premium: {down_out_model.derivative.premium}")
# %%
F = 1000
k = 1.4
X = 1.4
u = 1.6 / 1.4
d = 1.3 / 1.4
T = 1
Rd = 1.03
Rf = 1.05
model = ForexOptionSolver(expire=T, X=X, F=F, Rf=Rf, Rd=Rd, k=k, u=u, d=d, type="put", style="european")
print(f"Question 8 premium: {model.derivative.premium}")

# %%
