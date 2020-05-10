from scipy.special import jnyn_zeros


def alpha(l, n):
    """
    returns the `n`th positive root of the ordinary Bessel function of the first kind,
    J, of degree `l`
    """
    return jnyn_zeros(l, n)[0][-1]


def sort_alpha(i_max, l_max=100, n_max=100):
    alphas_ = []
    lns = []
    for n in range(1, n_max):
        for l in range(l_max):
            lns.append((l, n))
            alphas_.append(alpha(l, n))
    sorted_alphas = [ln for _, ln in sorted(zip(alphas_, lns))]
    for i, ln in enumerate(sorted_alphas):
        print(alpha(*ln), ',')
        if i == i_max:
            break
    print()
    print()
    for i, ln in enumerate(sorted_alphas):
        print(ln, ',')
        if i == i_max:
            break


if __name__ == '__main__':
    # carrier_test(Eg=1.42*q)
    # coefficients_test(Ef='intrinsic')
    # bulk_zT_test(2/3)
    sort_alpha(50, 100, 100)
