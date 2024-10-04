# %% Question 1
from src.solver import AssetOptionSolver, BarrierAssetOptionSolver, ForexOptionSolver

K = 26
T = 3
S = 26
u = 1.2
d = 1 / u
R = 1.05
B = 21

european_put = AssetOptionSolver(expire=T, S=S, u=u, d=d, K=K, type="put", style="european", R=R)

print(f"European put premium: {european_put.derivative.premium}")

american_put = AssetOptionSolver(expire=T, S=S, u=u, d=d, K=K, type="put", style="american", R=R)
print(f"American put premium: {american_put.derivative.premium}")

european_down_out = BarrierAssetOptionSolver(expire=T, S=S, u=u, d=d, K=K, type="put", style="european", R=R, barrier_type="down and out", B=B)
print(f"European down and out premium: {european_down_out.derivative.premium}")

american_down_out = BarrierAssetOptionSolver(expire=T, S=S, u=u, d=d, K=K, type="put", style="american", R=R, barrier_type="down and out", B=B)
print(f"American down and out premium: {american_down_out.derivative.premium}")

# %% Question 3

from src.solver import ForexOptionSolver

T = 3
X = 1 / 96.65
u = 1.06
d = 0.94
Rd = 1.08
Rf = 1.09
F = 1
k = 0.01080

call = ForexOptionSolver(expire=T, X=X, F=F, Rf=Rf, Rd=Rd, k=k, u=u, d=d, type="call", style="european")
put = ForexOptionSolver(expire=T, X=X, F=F, Rf=Rf, Rd=Rd, k=k, u=u, d=d, type="put", style="european")
print("Call premium: ", call.derivative.premium)
print("Put premium: ", put.derivative.premium)
