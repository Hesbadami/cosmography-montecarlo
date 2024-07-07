import sympy as sym

t, q, j, s, l, m, n = sym.symbols('t q j s l m n')

a = sym.Function('a')
H = sym.Function('H')

h = sym.diff(a(t), t) / a(t)
params = [1, -q, j, s, l, m, n]


def dHdt(order):
    dHdt = []
    # dH/dt
    for i in range(order):
        hd = sym.diff(h, t, i+1)

        for j in range(i+2, 0, -1):
            hd = hd.subs(
                (sym.diff(a(t), t, j)), params[j-1]*a(t)*(H(t)**j)
            )
        hd = (hd / H(t)**(i+2)).simplify()
        dHdt.append(hd)
    return dHdt


def dHdz(order):
    dHdz = []
    # dH/dz = -a/H * dH/dt
    r = H(t)
    for i in range(order):
        r = -a(t)/H(t) * sym.diff(r, t)
        r = r.subs((sym.diff(a(t), t)), a(t)*H(t))
        dHdz.append(r)

    for j, Hubble in enumerate(dHdz):
        order = j+1
        for i in range(order, 0, -1):
            Hubble = Hubble.subs(
                (sym.diff(H(t), t, i)),
                dHdt(order)[i-1]*(H(t)**(i+1))
            )
        Hubble = Hubble.expand().factor(H(t))/(H(t)*a(t)**order)
        dHdz[j] = Hubble
    return dHdz


if __name__ == "__main__":
    order = 4
    with open(
        'taylor_coeffs.txt', 'w'
    ) as log:
        for i, dh in enumerate(dHdt(order)):
            log.write(f'H{"d"*(i+1)}ot/H^{i+2} = {dh}\n')
        log.write('\n')
        for i, dh in enumerate(dHdz(order)):
            log.write(f'H_{i+1} = {dh}\n')
