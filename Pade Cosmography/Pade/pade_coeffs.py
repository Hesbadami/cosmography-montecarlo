# Code to compute Pade coefficients p_k and q_k
import sympy as sym

T = 4  # Expansion Terms
p1, p2 = 2, 2  # Pade dimensions

z = sym.Symbol('z')  # Redshift symbol
H = (1,) + sym.symbols(f'H_1:{T+1}')  # Taylor coefficients


# Populating Taylor Expansion
E = 0
for i in range(T+1):
    E += H[i] * (z**i / sym.factorial(i))


# Pade Approximation
def Pade(n, m):
    global P, Q
    P = sym.symbols(f'P_0:{n+1}')
    Q = sym.symbols(f'Q_0:{m+1}')
    num = 0
    denom = 1

    for i in range(n+1):
        # Populating numerator
        num += P[i]*(z**i)

    for i in range(1, m+1):
        # Populating denominator
        denom += Q[i]*(z**i)

    return num, denom


pad = E*Pade(p1, p2)[1]  # Multiplying Pade denominator by LHS
LHS = sym.Poly(pad, z)  # Creating a polynomial object
RHS = sym.Poly(Pade(p1, p2)[0], z)
ps = sym.solve(
    [
        sym.Eq(
            LHS.all_coeffs()[::-1][i],
            RHS.all_coeffs()[::-1][i]
        )
        for i in range(p1+1)
    ],
    P
)

# ps = e.all_coeffs()[::-1][:p1+1]

qs = sym.solve(
    [
        sym.Eq(
            LHS.all_coeffs()[::-1][i],
            0
        )
        for i in range(p1+1, p1+p2+1)
    ],
    Q
)

# qs = e.all_coeffs()[::-1][p1+1:p1+p2+1]

with open(f'pade_{p1}{p2}_coeffs.txt', 'w') as log:
    for i, p in enumerate(ps.values()):
        log.write(f'P_{i} = {p}\n')

    log.write('\n')

    for i, q in enumerate(qs.values()):
        log.write(f'Q_{i+1} = {q}\n')
