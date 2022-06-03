import matplotlib.pyplot as plt


def set_figure_params(tex_fonts=False, ):
    if tex_fonts:
        # tex configuration
        tex_fonts = {
            "text.usetex": True,
            "font.family": "serif",
            "axes.labelsize": 10,
            "font.size": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8
        }
        plt.rcParams.update(tex_fonts)
        plt.rcParams['text.latex.preamble'] = [r'\usepackage{underscore}']
