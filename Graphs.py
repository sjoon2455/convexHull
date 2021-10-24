from matplotlib import pyplot as plt
# Average times were gained by running all algorthims over all data
# sets 200 times and taking the average time


def graph_set_A(sizes):
    """
        Outputs a graph that compares all three algorthims over
        data set A
    """
    # The average times taken for all three algorithms over data set A in milli seconds
    gift_milli_A = [x*1000 for x in [0.007160457372665405, 0.014404278993606568, 0.022417218685150148, 0.04061191082000733, 0.0503708827495575,
                                     0.059285746812820436, 0.06884238004684448, 0.09941885828971862, 0.10758164763450623, 0.12186283707618713]]
    grah_milli_A = [x*1000 for x in [0.007544989585876465, 0.015188815593719483, 0.022944544553756715, 0.031321717500686644, 0.0395938789844513,
                                     0.0483058762550354, 0.05676034212112427, 0.06481868624687195, 0.07451822519302369, 0.08214217185974121]]
    mono_milli_A = [x*1000 for x in [0.009165838956832886, 0.018670244216918944, 0.02833400249481201, 0.038080735206604, 0.04824641227722168,
                                     0.059222618341445925, 0.06878257393836976, 0.07932158946990966, 0.08988933444023132, 0.10021942853927612]]
    incrmt_milli_A = [x*1000 for x in [0.035566091537475586, 0.06880378723144531, 0.11741971969604492, 0.16190195083618164,
                                       0.22816205024719238, 0.4328451156616211, 0.6385509967803955, 0.7811589241027832, 1.1505043506622314, 1.2333478927612305]]
    chan_milli_A = [x*1000 for x in [0.03403592109680176, 0.06840014457702637, 0.10280489921569824, 0.15466022491455078,
                                     0.19255805015563965, 0.2287290096282959, 0.2714848518371582, 0.4171741008758545, 0.46698594093322754, 0.5151829719543457]]
    quick_milli_A = [x*1000 for x in [0.02678084373474121, 0.12681913375854492, 0.10858702659606934, 0.14499115943908691,
                                      0.767829179763794, 0.12487316131591797, 0.13255691528320312, 1.616196870803833, 0.2182471752166748, 3.065459966659546]]
    kirk_seidel_milli_A = [x*1000 for x in [0.10343194007873535, 0.2851271629333496, 0.4528038501739502, 0.7949211597442627,
                                            1.3169498443603516, 2.773983955383301, 5.359673261642456, 6.660469055175781, 7.3340349197387695, 7.845739841461182]]
    dnq_milli_A = [x*1000 for x in [0.01417684555053711, 0.029835939407348633, 0.05237603187561035, 0.06327605247497559,
                                    0.09257864952087402, 0.09712600708007812, 0.12168192863464355, 0.1527271270751953, 0.1725449562072754, 0.17263388633728027]]
    melkman_milli_A = [x*1000 for x in [0.016674041748046875, 0.03496289253234863, 0.0537412166595459, 0.07932519912719727,
                                        0.09166908264160156, 0.11235308647155762, 0.1331651210784912, 0.1532742977142334, 0.1911330223083496, 0.2129349708557129]]
    btr_milli_A = [x*1000 for x in [0.018963098526000977, 0.038366079330444336, 0.06129002571105957, 0.07958698272705078,
                                    0.09939813613891602, 0.11249709129333496, 0.1389310359954834, 0.15305399894714355, 0.17258381843566895, 0.18652701377868652]]
    symm_milli_A = [x*1000 for x in [0.0018541812896728516, 0.01181793212890625, 0.018368005752563477, 0.011520862579345703,
                                     0.01638197898864746, 0.016000986099243164, 0.018296241760253906, 0.03674602508544922, 0.03046703338623047, 0.03770804405212402]]
    # graph showing average times against number of total points of data set A
    # plt.plot(sizes, gift_milli_A, label="Giftwrap", linestyle='--')
    # plt.plot(sizes, grah_milli_A, label="Grahamscan", linestyle='-.')
    # plt.plot(sizes, mono_milli_A, label="Monotone chain", linestyle=':')
    # plt.plot(sizes, incrmt_milli_A, label="Incremental", linestyle='-')
    # plt.plot(sizes, chan_milli_A, label="Chan", color="blue")
    # plt.plot(sizes, quick_milli_A, label="Quick", color="red")
    # plt.plot(sizes, kirk_seidel_milli_A, label="KS", color="black")
    # plt.plot(sizes, dnq_milli_A, label="DnQ", color="green")
    # plt.plot(sizes, melkman_milli_A, label="Melkman", color="yellow")
    # plt.plot(sizes, btr_milli_A, label="BinaryTree", color="magenta")
    # plt.plot(sizes, symm_milli_A, label="Symmetric", color="cyan")
    plt.plot(sizes, gift_milli_A,
             color="grey")
    plt.plot(sizes, grah_milli_A,
             color="grey")
    plt.plot(sizes, mono_milli_A,
             color="grey")
    plt.plot(sizes, incrmt_milli_A,
             color="grey")
    plt.plot(sizes, chan_milli_A,  color="grey")
    plt.plot(sizes, kirk_seidel_milli_A,  color="grey")
    plt.plot(sizes, dnq_milli_A,  color="grey")
    plt.plot(sizes, melkman_milli_A,  color="grey")
    plt.plot(sizes, btr_milli_A, label="BinaryTree", color="red")
    plt.plot(sizes, symm_milli_A, label="Symmetric", color="blue")

    from matplotlib.lines import Line2D
    line = Line2D([0], [0], label='Other Algorithms', color='grey')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([line])

    plt.legend(handles=handles)
    plt.xlabel("Number of input points")
    plt.ylabel("Time (ms)")
    plt.grid(color='b', linestyle='-', linewidth=.1)
    # plt.legend(loc=2)
    plt.xticks(sizes)
    plt.show()


def graph_set_B(sizes):
    """
        Outputs a graph that compares all three algorthims over
        data set B
    """
    # The average times taken for all three algorithms over data set B in milli seconds
    gift_milli_A = [x*1000 for x in [0.054100990295410156, 0.20097017288208008, 0.44912099838256836, 0.8075499534606934,
                                     1.2640039920806885, 1.8008768558502197, 2.450896978378296, 3.208796977996826, 4.048182010650635, 5.097496032714844]]
    grah_milli_A = [x*1000 for x in [0.006411075592041016, 0.01186513900756836, 0.01765894889831543, 0.023797035217285156,
                                     0.029566049575805664, 0.03693795204162598, 0.04221224784851074, 0.04843711853027344, 0.05597329139709473, 0.06205892562866211]]
    mono_milli_A = [x*1000 for x in [0.00616908073425293, 0.012781143188476562, 0.01866602897644043, 0.02460503578186035,
                                     0.031980037689208984, 0.037867069244384766, 0.04472208023071289, 0.05114579200744629, 0.057882070541381836, 0.0661001205444336]]
    incrmt_milli_A = [x*1000 for x in [0.03337407112121582, 0.06975388526916504, 0.11698675155639648, 0.1674351692199707,
                                       0.24362802505493164, 0.36735105514526367, 0.5192320346832275, 0.7505819797515869, 1.0499298572540283, 1.4304230213165283]]
    chan_milli_A = [x*1000 for x in [0.07366394996643066, 0.1556720733642578, 0.24401497840881348, 0.3463780879974365,
                                     0.44069695472717285, 0.5503301620483398, 0.6774699687957764, 0.7950839996337891, 0.9920499324798584, 1.1135399341583252]]
    quick_milli_A = [x*1000 for x in [0.02664804458618164, 0.05340695381164551, 0.0837697982788086, 0.11192512512207031,
                                      0.1427137851715088, 0.17136073112487793, 0.20420384407043457, 0.2244870662689209, 0.26059794425964355, 0.2922382354736328]]
    kirk_seidel_milli_A = [x*1000 for x in [0.15878510475158691, 0.4971578121185303, 1.012800931930542, 1.8148159980773926,
                                            2.782431125640869, 4.542675018310547, 6.734553098678589, 9.400645017623901, 12.156552076339722, 15.719734907150269]]
    # dnq_milli_A = [x*1000 for x in [0.01562190055847168]]
    melkman_milli_A = [x*1000 for x in [0.01699686050415039, 0.03548693656921387, 0.05443716049194336, 0.07398414611816406,
                                        0.1018829345703125, 0.12133407592773438, 0.14424800872802734, 0.15377092361450195, 0.17800498008728027, 0.21046900749206543]]
    btr_milli_A = [x*1000 for x in [0.019087791442871094, 0.038484811782836914, 0.05751228332519531, 0.09172987937927246,
                                    0.09641194343566895, 0.11378169059753418, 0.14107275009155273, 0.15179109573364258, 0.17624711990356445, 0.19264793395996094]]
    symm_milli_A = [x*1000 for x in [0.003865957260131836, 0.007860183715820312, 0.011834859848022461, 0.01576519012451172,
                                     0.01980900764465332, 0.023828983306884766, 0.028023958206176758, 0.03180289268493652, 0.03620576858520508, 0.04076099395751953]]
    # graph showing average times against number of total points of data set B
    plt.plot(sizes, gift_milli_A,
             color="grey")
    plt.plot(sizes, grah_milli_A,
             color="grey")
    # plt.plot(sizes, mono_milli_A,
    #  color="grey")
    plt.plot(sizes, incrmt_milli_A,
             color="grey")
    plt.plot(sizes, chan_milli_A,  color="grey")
    # plt.plot(sizes, kirk_seidel_milli_A,  color="grey")
    plt.plot(sizes, melkman_milli_A,  color="grey")
    plt.plot(sizes, btr_milli_A, label="BinaryTree", color="red")
    plt.plot(sizes, symm_milli_A, label="Symmetric", color="blue")

    from matplotlib.lines import Line2D
    line = Line2D([0], [0], label='Other Algorithms', color='grey')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([line])
    plt.legend(handles=handles)

    plt.xlabel("Number of input points")
    plt.ylabel("Time (ms)")
    plt.grid(color='b', linestyle='-', linewidth=.1)
    plt.xticks(sizes)
    plt.show()


# def graph_all(sizes):
#     """Compares all algorithms over all data sets against input size
#     """
#     # The average times for the gift wrap algorithm in milli seconds
#     gift_milli_B = [x*1000 for x in [0.06978759288787842, 0.2801102638244629, 0.6163508462905883, 1.0805705881118775, 1.7255639696121217,
#                                      2.4955154180526735, 3.370841302871704, 4.517650356292725, 5.745762286186218, 7.038509984016418]]
#     gift_milli_A = [x*1000 for x in [0.007160457372665405, 0.014404278993606568, 0.022417218685150148, 0.04061191082000733, 0.0503708827495575,
#                                      0.059285746812820436, 0.06884238004684448, 0.09941885828971862, 0.10758164763450623, 0.12186283707618713]]
#     # The average times for the Graham-scan algorithm in milli seconds
#     grah_milli_B = [x*1000 for x in [0.007873213291168213, 0.016229190826416016, 0.02476203203201294, 0.033674232959747315, 0.042335739135742186,
#                                      0.050941919088363645, 0.05983875155448914, 0.06785545945167541, 0.07610330700874329, 0.08472154974937439]]
#     grah_milli_A = [x*1000 for x in [0.007544989585876465, 0.015188815593719483, 0.022944544553756715, 0.031321717500686644, 0.0395938789844513,
#                                      0.0483058762550354, 0.05676034212112427, 0.06481868624687195, 0.07451822519302369, 0.08214217185974121]]
#     # The average times for the Monotone chain algorithm in milli seconds
#     mono_milli_B = [x*1000 for x in [0.009106448888778686, 0.01869452953338623, 0.02827052116394043, 0.038180583715438844, 0.04822150468826294,
#                                      0.05885208964347839, 0.06831400275230408, 0.07893061757087708, 0.08912485957145691, 0.10061038613319397]]
#     mono_milli_A = [x*1000 for x in [0.009165838956832886, 0.018670244216918944, 0.02833400249481201, 0.038080735206604, 0.04824641227722168,
#                                      0.059222618341445925, 0.06878257393836976, 0.07932158946990966, 0.08988933444023132, 0.10021942853927612]]
#     # graph showing average times against number of total points over all data sets
#     plt.plot(sizes, gift_milli_A, label="Giftwrap - Set_A",
#              linestyle='--', color="blue")
#     # plt.plot(sizes, gift_milli_B, label="Giftwrap - Set_B")
#     plt.plot(sizes, grah_milli_A, label="Graham-scan - Set_A",
#              linestyle='--', color="red")
#     plt.plot(sizes, grah_milli_B, label="Graham-scan - Set_B", color="red")
#     plt.plot(sizes, mono_milli_A, label="Monotone chain - Set_A",
#              linestyle='--', color="fuchsia")
#     plt.plot(sizes, mono_milli_B,
#              label="Monotone chain - Set_B", color="fuchsia")
#     plt.xlabel("Number of input points")
#     plt.ylabel("Time (ms)")
#     plt.grid(color='b', linestyle='-', linewidth=.1)
#     plt.legend(loc=2)
#     plt.xticks(sizes)
#     plt.show()


# def graph_gift(sizes):
#     """
#         Outputs a grpah that shows the average time of the Gift wrap algorithm
#         over both data sets
#     """
#     # The average times taken over data set A in milli seconds
#     gift_milli_A = [x*1000 for x in [0.007160457372665405, 0.014404278993606568, 0.022417218685150148, 0.04061191082000733, 0.0503708827495575,
#                                      0.059285746812820436, 0.06884238004684448, 0.09941885828971862, 0.10758164763450623, 0.12186283707618713]]
#     # The average times taken over data set B in milli seconds
#     gift_milli_B = [x*1000 for x in [0.06978759288787842, 0.2801102638244629, 0.6163508462905883, 1.0805705881118775, 1.7255639696121217,
#                                      2.4955154180526735, 3.370841302871704, 4.517650356292725, 5.745762286186218, 7.038509984016418]]
#     # Graph comparing the times taken in data set A and B for the gift wrap algorithm
#     plt.plot(sizes, gift_milli_A, label="Data Set A", linestyle='-.')
#     plt.plot(sizes, gift_milli_B, label="Data Set B", linestyle='--')
#     plt.xlabel("Number of input points")
#     plt.ylabel("Time (ms)")
#     plt.grid(color='b', linestyle='-', linewidth=.1)
#     plt.legend(loc=2)
#     plt.xticks(sizes)
#     plt.show()


# def graph_graham(sizes):
#     """
#         Outputs a grpah that shows the average time of the Graham-scan algorithm
#         over both data sets
#     """
#     # The average times taken over data set A
#     grah_milli_A = [x*1000 for x in [0.007544989585876465, 0.015188815593719483, 0.022944544553756715, 0.031321717500686644, 0.0395938789844513,
#                                      0.0483058762550354, 0.05676034212112427, 0.06481868624687195, 0.07451822519302369, 0.08214217185974121]]
#     # The average times taken over data set B
#     grah_milli_B = [x*1000 for x in [0.007873213291168213, 0.016229190826416016, 0.02476203203201294, 0.033674232959747315, 0.042335739135742186,
#                                      0.050941919088363645, 0.05983875155448914, 0.06785545945167541, 0.07610330700874329, 0.08472154974937439]]
#     # Graph comparing the times taken in data set A and B for the Graham-scan algorithm
#     plt.plot(sizes, grah_milli_A, label="Data Set A", linestyle='-.')
#     plt.plot(sizes, grah_milli_B, label="Data Set B", linestyle='--')
#     plt.xlabel("Number of input points")
#     plt.ylabel("Time (ms)")
#     plt.grid(color='b', linestyle='-', linewidth=.1)
#     plt.legend(loc=2)
#     plt.xticks(sizes)
#     plt.show()


# def graph_mono(sizes):
#     """
#         Outputs a graph that shows the average time of the Monotone chain algorithm
#         over both data sets
#     """
#     # The average times for the Monotone chan in milli seconds
#     mono_milli_B = [x*1000 for x in [0.009106448888778686, 0.01869452953338623, 0.02827052116394043, 0.038180583715438844, 0.04822150468826294,
#                                      0.05885208964347839, 0.06831400275230408, 0.07893061757087708, 0.08912485957145691, 0.10061038613319397]]
#     mono_milli_A = [x*1000 for x in [0.009165838956832886, 0.018670244216918944, 0.02833400249481201, 0.038080735206604, 0.04824641227722168,
#                                      0.059222618341445925, 0.06878257393836976, 0.07932158946990966, 0.08988933444023132, 0.10021942853927612]]
#     # Graph comparing the times taken in data set A and B for the Monotone chain algorithm
#     plt.plot(sizes, mono_milli_A, label="Data Set A", linestyle='--')
#     plt.plot(sizes, mono_milli_B, label="Data Set B", linestyle='-.')
#     plt.xlabel("Number of input points")
#     plt.ylabel("Time (ms)")
#     plt.grid(color='b', linestyle='-', linewidth=.1)
#     plt.legend(loc=2)
#     plt.xticks(sizes)
#     plt.show()


# def graph_incrmt(sizes):
#     """
#         Outputs a graph that shows the average time of the Monotone chain algorithm
#         over both data sets
#     """
#     # The average times for the Monotone chan in milli seconds
#     mono_milli_B = [x*1000 for x in [0.009106448888778686, 0.01869452953338623, 0.02827052116394043, 0.038180583715438844, 0.04822150468826294,
#                                      0.05885208964347839, 0.06831400275230408, 0.07893061757087708, 0.08912485957145691, 0.10061038613319397]]
#     mono_milli_A = [x*1000 for x in [0.009165838956832886, 0.018670244216918944, 0.02833400249481201, 0.038080735206604, 0.04824641227722168,
#                                      0.059222618341445925, 0.06878257393836976, 0.07932158946990966, 0.08988933444023132, 0.10021942853927612]]
#     # Graph comparing the times taken in data set A and B for the Monotone chain algorithm
#     plt.plot(sizes, mono_milli_A, label="Data Set A", linestyle='--')
#     plt.plot(sizes, mono_milli_B, label="Data Set B", linestyle='-.')
#     plt.xlabel("Number of input points")
#     plt.ylabel("Time (ms)")
#     plt.grid(color='b', linestyle='-', linewidth=.1)
#     plt.legend(loc=2)
#     plt.xticks(sizes)
#     plt.show()


# def graph_all_hull(sizes_A, sizes_B):
#     """Compares all algorithms over all data sets against convex hull size
#     """
#     # The average times for the gift wrap algorithm in milli seconds
#     gift_milli_B = [x*1000 for x in [0.06978759288787842, 0.2801102638244629, 0.6163508462905883, 1.0805705881118775, 1.7255639696121217,
#                                      2.4955154180526735, 3.370841302871704, 4.517650356292725, 5.745762286186218, 7.038509984016418]]
#     gift_milli_A = [x*1000 for x in [0.007160457372665405, 0.014404278993606568, 0.022417218685150148, 0.04061191082000733, 0.0503708827495575,
#                                      0.059285746812820436, 0.06884238004684448, 0.09941885828971862, 0.10758164763450623, 0.12186283707618713]]
#     # The average times for the Graham-scan algorithm in milli seconds
#     grah_milli_B = [x*1000 for x in [0.007873213291168213, 0.016229190826416016, 0.02476203203201294, 0.033674232959747315, 0.042335739135742186,
#                                      0.050941919088363645, 0.05983875155448914, 0.06785545945167541, 0.07610330700874329, 0.08472154974937439]]
#     grah_milli_A = [x*1000 for x in [0.007544989585876465, 0.015188815593719483, 0.022944544553756715, 0.031321717500686644, 0.0395938789844513,
#                                      0.0483058762550354, 0.05676034212112427, 0.06481868624687195, 0.07451822519302369, 0.08214217185974121]]
#     # The average times for the Monotone chain algorithm in milli seconds
#     mono_milli_B = [x*1000 for x in [0.009106448888778686, 0.01869452953338623, 0.02827052116394043, 0.038180583715438844, 0.04822150468826294,
#                                      0.05885208964347839, 0.06831400275230408, 0.07893061757087708, 0.08912485957145691, 0.10061038613319397]]
#     mono_milli_A = [x*1000 for x in [0.009165838956832886, 0.018670244216918944, 0.02833400249481201, 0.038080735206604, 0.04824641227722168,
#                                      0.059222618341445925, 0.06878257393836976, 0.07932158946990966, 0.08988933444023132, 0.10021942853927612]]
#     fig, ax1 = plt.subplots()
#     # graph showing average times against number of points in the convex hull over both data sets
#     ax1.plot(sizes_A, gift_milli_A,
#              label="Giftwrap - Set_A", color="fuchsia")
#     # plt.plot(sizes, gift_milli_B, label="Giftwrap - Set_B")
#     ax1.plot(sizes_A, grah_milli_A, label="Graham-scan - Set_A",
#              linestyle='-.', color="fuchsia")
#     ax1.plot(sizes_A, mono_milli_A, label="Monotone chain - Set_A",
#              linestyle='--', color="fuchsia")
#     ax1.set_xlabel("Number of convex hull points in data set A",
#                    color="fuchsia")
#     ax1.set_ylabel("Time (ms)")
#     ax1.set_xticks(sizes_A)
#     ax1.tick_params(axis='x', labelcolor="fuchsia")
#     # Adding second x axis
#     ax2 = ax1.twiny()
#     ax2.set_xlabel("Number of convex hull points in data set B", color="red")
#     ax2.plot(sizes_B, mono_milli_B, label="Monotone chain - Set_B",
#              linestyle='--', color="red")
#     ax2.plot(sizes_B, grah_milli_B, label="Graham-scan - Set_B",
#              linestyle='-.', color="red")
#     ax2.set_xticks(sizes_B)
#     ax2.tick_params(axis='x', labelcolor="red")
#     plt.grid(color='b', linestyle='-', linewidth=.1)
#     ax1.legend(loc=2)
#     ax2.legend(loc=4)
#     fig.tight_layout()
#     plt.show()
