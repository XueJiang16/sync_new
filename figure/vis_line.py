# import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data_auroc = '''66.44
                    64.42
                    54.34
                    62.12
                    68.46
                    66.28
                    69.75
                    66.83
                    66.31
                    53.13
                    64.62
                    72.08
                    70.45
                    73.59
                    66.44
                    65.59
                    53.13
                    63.55
                    70.81
                    69.33
                    72.93
                    65.89
                    66.26
                    53.41
                    64.68
                    71.91
                    70.00
                    74.53
                    64.65
                    65.13
                    56.59
                    63.63
                    70.60
                    68.56
                    73.21
                    61.49
                    61.73
                    52.71
                    60.47
                    69.74
                    68.58
                    73.91
                    60.03
                    60.69
                    55.55
                    59.93
                    71.26
                    71.31
                    76.05
                    '''
    data_fpr = '''87.95
                    88.21
                    92.77
                    89.65
                    77.84
                    79.42
                    76.72
                    88.37
                    88.28
                    91.39
                    89.72
                    74.01
                    75.71
                    72.07
                    89.16
                    89.63
                    92.96
                    91.44
                    77.33
                    78.58
                    74.32
                    90.36
                    90.63
                    90.58
                    91.26
                    77.11
                    78.40
                    72.42
                    90.50
                    91.36
                    88.39
                    92.20
                    78.92
                    80.76
                    74.92
                    94.23
                    94.43
                    92.70
                    94.85
                    81.57
                    80.68
                    74.50
                    95.44
                    95.52
                    92.78
                    95.30
                    80.45
                    76.11
                    70.12
                    '''



    # data_list = list(map(float, data_auroc.split()))
    data_list = list(map(float, data_fpr.split()))
    msp = data_list[::7]
    odin = data_list[1::7]
    maha = data_list[2::7]
    energy = data_list[3::7]
    gradnorm = data_list[4::7]
    dice = data_list[5::7]
    ours = data_list[6::7]
    from matplotlib.ticker import MaxNLocator



    y = ['{}'.format(i) for i in range(2,9)]
    # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig=plt.figure()
    ax = fig.add_subplot(111)
    ax.patch.set_facecolor('lightsteelblue')
    ax.patch.set_alpha(0.1)

    plt.plot(y, msp, label='MSP', marker="d")
    plt.plot(y, odin, label='ODIN', marker="x")
    plt.plot(y, maha, label='Mahalanobis', marker="o")
    plt.plot(y, energy, label='Energy', marker="p")
    plt.plot(y, gradnorm, label='GradNorm', marker="v")
    plt.plot(y, dice, label='Dice', marker="^")
    plt.plot(y, ours, label='RP+GradNorm(Ours)', marker="*")
    plt.legend(loc='upper left', fontsize=14, ncol=2, framealpha=0.5)
    plt.xlabel('Tail Index a', fontsize=18, fontweight='bold')
    plt.ylabel('Average FPR95 (%)', fontsize=18, fontweight='bold')
    # plt.ylim((50,90))
    plt.ylim((69, 103))
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.grid(color='w')
    ax = plt.gca()  # 获取当前的axes
    ax.spines['right'].set_color('w')
    ax.spines['top'].set_color('w')
    ax.spines['left'].set_color('w')
    ax.spines['bottom'].set_color('w')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    # plt.show()
    plt.savefig('line_fpr.pdf')

