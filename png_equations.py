import latexmath2png

# A quick guide to tex equations
"""def math2png(eqs, outdir, packages = default_packages, prefix = '', size = 1):
    Parameters:
	eqs         - A list of equations
	outdir      - Output directory for PNG images
	packages    - Optional list of packages to include in the LaTeX preamble
	prefix      - Optional prefix for output files
	size        - Scale factor for output"""

pre = 'gtktex_'
eqs = [
    r'$\alpha_i > \beta_i$',
    r'$\sum_{i=0}^\infty x_i$',
    r'$\left(\frac{5 - \frac{1}{x}}{4}\right)$',
    r'$s(t) = \mathcal{A}\sin(2 \omega t)$',
    r'$\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$'
    ]
latexmath2png.math2png(eqs, "./equations/", prefix = pre)
latexmath2png.math2png([r'Y = c + \alpha~D + \beta~G(x)~D^2'], "./equations/", prefix = "YfuncG", size=1.5)
latexmath2png.math2png([r'Y = c + \alpha~D + \beta~D^2'], "./equations/", prefix = "Yfunc", size=1.5)
latexmath2png.math2png([r'$G(x) = \frac{2}{x^2}~\left( x-1+e^{-x} \right),~~~x=\frac{t}{t_0}$'], "./equations/", prefix = "Gfunc", size=1.3)


