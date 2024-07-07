# Code to compute rational chebyshev pade approximation coefficients
import sympy as sym

order = 4  # Expansion Terms
p1, p2 = 2, 2  # Pade dimensions

z = sym.Symbol('z')  # Redshift symbol
c = sym.symbols(f'c_0:{order+1}')  # Chebyshev coefficients


# Chebyshev polynomials through recursion relation
def T(n):
    Ts = [1, z]

    if n in (0, 1):
        return Ts[n]

    else:
        k = 2
        while k <= n:
            Tk = 2*z*Ts[-1] - Ts[-2]
            Ts.append(Tk)
            k += 1
        return Tk.expand()


# Populating Chebyshev Expansion
E = 0
for i in range(order+1):
    E += c[i] * T(i)


# Rational Chebyshev Pade approximation
def Ch_Pade(n, m):
    global A, B
    A = sym.symbols(f'A_0:{n+1}')
    B = sym.symbols(f'B_0:{m+1}')
    num = 0
    denom = 1

    for i in range(n+1):
        # Populating numerator
        num += A[i] * T(i)

    for i in range(1, m+1):
        # Populating denominator
        denom += B[i] * T(i)

    return num, denom


ch_pad = E*Ch_Pade(p1, p2)[1]  # Multiplying Pade denominator by LHS
LHS = sym.Poly(ch_pad, z)
RHS = sym.Poly(Ch_Pade(p1, p2)[0], z)

As = sym.solve(
    [
        sym.Eq(
            LHS.all_coeffs()[::-1][i],
            RHS.all_coeffs()[::-1][i]
        )
        for i in range(p1+1)
    ],
    A
)

Bs = sym.solve(
    [
        sym.Eq(
            LHS.all_coeffs()[::-1][i],
            0
        )
        for i in range(p1+1, p1+p2+1)
    ],
    B
)

with open(f'chebyshev_pade_{p1}{p2}_coeffs.txt', 'w') as log:
    for i, a in enumerate(As.values()):
        log.write(f'A_{i} = {a}\n')

    log.write('\n')

    for i, b in enumerate(Bs.values()):
        log.write(f'B_{i+1} = {b}\n')
