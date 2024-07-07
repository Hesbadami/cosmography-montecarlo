# code to compute c_i (chebyshev expansion coefficients)
import sympy as sym

order = 4

q, j, s, l, m, n = sym.symbols('q j s l m n')
H = (1,) + sym.symbols(f'H_1:{order+1}')  # Taylor coefficients

z = sym.Symbol('z')


# Truncated Taylor Expansion
def g(order):
    g = 0
    for i in range(order+1):
        g += H[i] * (z**(i) / sym.factorial(i))

    return g


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


def c(n):
    omega = 1/sym.sqrt(1 - z**2)
    if n == 0:
        return (
            1/sym.pi * sym.integrate(
                g(order) * T(n) * omega, (z, -1, 1)
            )
        )
    else:
        return (
            2/sym.pi * sym.integrate(
                g(order) * T(n) * omega, (z, -1, 1)
            )
        )


with open('chebyshev_coeffs.txt', 'w') as log:
    for n in range(order+1):
        log.write(f'c_{n} = {c(n).expand()}\n')
    log.write('\n')
    for n in range(order+1):
        log.write(f'T_{n} = {T(n)}\n')
