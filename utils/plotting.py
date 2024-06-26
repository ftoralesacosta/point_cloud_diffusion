import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mtick

colors = {
    'true': 'black',
    'gen': '#7570b3',
    'Geant': 'black',
    'GSGM': '#7570b3',

    'P_truth': '#7570b3',
    'P_gen': '#7570b3',
    'Theta_truth': '#d95f02',
    'Theta_gen': '#d95f02',
}

line_style = {
    'true': 'dotted',
    'gen': '-',
    'Geant': 'dotted',
    'GSGM': '-',
    'P_truth': '-',
    'P_gen': 'dotted',
    'Theta_truth': '-',
    'Theta_gen': 'dotted',
}

name_translate = {
    'true': 'True distribution',
    'gen': 'Generated distribution',
    'Geant': 'Geant 4',
    'GSGM': 'Graph Diffusion',

    'P_truth': 'Sim.: P',
    'P_gen': 'FPCD: P',
    'Theta_truth': 'Sim.: Theta',
    'Theta_gen': 'FPCD: Theta',

    # 't_gen_d64': 'FPCD: top 8 steps',
    # 't_gen_d256': 'FPCD: top 2 steps',
    }


def SetStyle():
    from matplotlib import rc
    rc('text', usetex=True)

    import matplotlib as mpl
    rc('font', family='serif')
    rc('font', size=22)
    rc('xtick', labelsize=15)
    rc('ytick', labelsize=15)
    rc('legend', fontsize=15)

    # #
    mpl.rcParams.update({'font.size': 19})
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams.update({'xtick.labelsize': 18})
    mpl.rcParams.update({'ytick.labelsize': 18})
    mpl.rcParams.update({'axes.labelsize': 18})
    mpl.rcParams.update({'legend.frameon': False})
    mpl.rcParams.update({'lines.linewidth': 2})

    import mplhep as hep
    hep.style.use("CMS")


def SetGrid(ratio=True):
    fig = plt.figure(figsize=(9, 9))
    if ratio:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        gs.update(wspace=0.025, hspace=0.1)
    else:
        gs = gridspec.GridSpec(1, 1)
    return fig,gs


def PlotRoutine(feed_dict, xlabel='', ylabel='', reference_name='gen'):
    assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"

    fig, gs = SetGrid()
    ax0 = plt.subplot(gs[0])
    plt.xticks(fontsize=0)
    ax1 = plt.subplot(gs[1], sharex=ax0)

    for ip, plot in enumerate(feed_dict.keys()):

        if 'steps' in plot or 'r=' in plot:
            ax0.plot(np.mean(feed_dict[plot], 0), label=plot, marker=line_style[plot], color=colors[plot], lw=0)

        else:
            ax0.plot(np.mean(feed_dict[plot], 0), label=plot, linestyle=line_style[plot], color=colors[plot])

        if reference_name != plot:
            ratio = 100*np.divide(np.mean(feed_dict[reference_name], 0)-np.mean(feed_dict[plot], 0),
                                  np.mean(feed_dict[reference_name], 0))

            if 'steps' in plot or 'r=' in plot:
                ax1.plot(ratio, color=colors[plot], markeredgewidth=1, marker=line_style[plot], lw=0)

            else:
                ax1.plot(ratio,color=colors[plot],linewidth=2,linestyle=line_style[plot])


    FormatFig(xlabel = "",  ylabel = ylabel, ax0=ax0)
    ax0.legend(loc='best', fontsize=16, ncol=1)

    plt.ylabel('Difference. (%)')
    plt.xlabel(xlabel)
    plt.axhline(y=0.0,  color='r',  linestyle='--', linewidth=1)
    plt.axhline(y=10,  color='r',  linestyle='--', linewidth=1)
    plt.axhline(y=-10,  color='r',  linestyle='--', linewidth=1)
    plt.ylim([-100, 100])

    return fig, ax0


class ScalarFormatterClass(mtick.ScalarFormatter):
    #https://www.tutorialspoint.com/show-decimal-places-and-scientific-notation-on-the-axis-of-a-matplotlib-plot
    def _set_format(self):
        self.format = "%1.1f"


def FormatFig(xlabel, ylabel, ax0):
    # Limit number of digits in ticks
    # y_loc, _ = plt.yticks()
    # y_update = ['%.1f' % y for y in y_loc]
    # plt.yticks(y_loc, y_update)
    ax0.set_xlabel(xlabel, fontsize=20)
    ax0.set_ylabel(ylabel)

    # xposition = 0.9
    # yposition=1.03
    # text = 'H1'
    # WriteText(xposition,yposition,text,ax0)


def WriteText(xpos, ypos, text, ax0):

    plt.text(xpos, ypos, text,
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax0.transAxes, fontsize=25, fontweight='bold')


def HistRoutine(feed_dict,
                xlabel='', ylabel='',
                reference_name='Geant',
                logy=False, binning=None,
                fig=None, gs=None,
                plot_ratio=True,
                idx=None,
                label_loc='best'):

    assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"

    if fig is None:
        fig, gs = SetGrid(plot_ratio)
    ax0 = plt.subplot(gs[0])
    if plot_ratio:
        plt.xticks(fontsize=0)
        ax1 = plt.subplot(gs[1], sharex=ax0)

    if binning is None:
        binning = np.linspace(np.quantile(feed_dict[reference_name], 0.0),
                              np.quantile(feed_dict[reference_name], 1), 5)

    xaxis = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]
    reference_hist, _ = np.histogram(feed_dict[reference_name], bins=binning, density=True)
    maxy = np.max(reference_hist) 
    print(reference_hist)  # [2.16 0.72 0.72 0.   0.   0.   0.   0.   0.  ]
    print(maxy)  # 2.1599999999999997

    for ip, plot in enumerate(feed_dict.keys()):

        # print("Plot",ip,": Shape of feed_dict = ",np.shape(feed_dict[plot]))
        # print("Feed Dict Keys = ",feed_dict.keys())
        # print("Name translate keys = ",name_translate.keys())
        # print("Line_style keys = ",line_style.keys())
        # print("Colors keys = ",colors.keys())

        dist, _, _ = ax0.hist(feed_dict[plot], bins=binning,
                              label=name_translate[plot],
                              linestyle=line_style[plot],
                              color=colors[plot],
                              density=True,
                              histtype="step")

        if plot_ratio:
            if reference_name != plot:
                ratio = 100*np.divide(reference_hist-dist, reference_hist)  # mark.
                ax1.plot(xaxis, ratio, color=colors[plot],
                         marker='o', ms=10, lw=0,
                         markerfacecolor='none',
                         markeredgewidth=3)

    ax0.legend(loc=label_loc, fontsize=12, ncol=5)

    if logy:
        ax0.set_yscale('log')

    if plot_ratio:
        FormatFig(xlabel="", ylabel=ylabel, ax0=ax0)
        plt.ylabel('Difference. (%)')
        plt.xlabel(xlabel)
        plt.axhline(y=0.0, color='r', linestyle='-', linewidth=1)
        # plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
        # plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
        plt.ylim([-100, 100])
    else:
        FormatFig(xlabel=xlabel, ylabel=ylabel, ax0=ax0)

    return fig, gs, binning
