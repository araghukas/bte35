"""
The energy eigenvalues for solutions of the Schrodinger equation in an infinite circular
well are proportional to the squares of the roots of ordinary Bessel functions of the 1st
kind.

The quantum numbers (l, n) imply the `n`th root of the order `l` Bessel function.

I don't know of a clever way to the order the `n`th roots across orders, so I've hard
coded the first 20 roots `alphas` and the corresponding quantum numbers `lns` below.
"""

alphas = [
    2.404825557695773,
    3.8317059702075125,
    5.135622301840683,
    5.520078110286311,
    6.380161895923984,
    7.015586669815619,
    7.5883424345038035,
    8.417244140399864,
    8.653727912911013,
    8.771483815959954,
    9.76102312998167,
    9.936109524217684,
    10.173468135062722,
    11.064709488501185,
    11.086370019245084,
    11.61984117214906,
    11.791534439014281,
    12.225092264004656,
    12.338604197466944,
    13.015200721698434,
]


lns = [
    (0, 1),
    (1, 1),
    (2, 1),
    (0, 2),
    (3, 1),
    (1, 2),
    (4, 1),
    (2, 2),
    (0, 3),
    (5, 1),
    (3, 2),
    (6, 1),
    (1, 3),
    (4, 2),
    (7, 1),
    (2, 3),
    (0, 4),
    (8, 1),
    (5, 2),
    (3, 3),
]
