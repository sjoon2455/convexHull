from matplotlib import pyplot as plt
#Average times were gained by running all algorthims over all data
#sets 200 times and taking the average time

def graph_set_A(sizes):
    """
        Outputs a graph that compares all three algorthims over
        data set A
    """
    #The average times taken for all three algorithms over data set A in milli seconds
    gift_milli_A = [x*1000 for x in [0.007160457372665405, 0.014404278993606568, 0.022417218685150148, 0.04061191082000733, 0.0503708827495575,
    0.059285746812820436, 0.06884238004684448, 0.09941885828971862, 0.10758164763450623, 0.12186283707618713]]
    grah_milli_A = [x*1000 for x in [0.007544989585876465, 0.015188815593719483, 0.022944544553756715, 0.031321717500686644, 0.0395938789844513,
    0.0483058762550354, 0.05676034212112427, 0.06481868624687195, 0.07451822519302369, 0.08214217185974121]]
    mono_milli_A = [x*1000 for x in [0.009165838956832886, 0.018670244216918944, 0.02833400249481201, 0.038080735206604, 0.04824641227722168,
    0.059222618341445925, 0.06878257393836976, 0.07932158946990966, 0.08988933444023132, 0.10021942853927612]]
    #graph showing average times against number of total points of data set A
    plt.plot(sizes, gift_milli_A, label="Giftwrap", linestyle='--')
    plt.plot(sizes, grah_milli_A, label="Grahamscan", linestyle='-.')
    plt.plot(sizes, mono_milli_A, label="Monotone chain")
    plt.xlabel("Number of input points")
    plt.ylabel("Time (ms)")
    plt.grid(color='b', linestyle='-', linewidth=.1)
    plt.legend(loc=2)
    plt.xticks(sizes)
    plt.show()

def graph_set_B(sizes):
    """
        Outputs a graph that compares all three algorthims over
        data set B
    """
    #The average times taken for all three algorithms over data set B in milli seconds
    gift_milli_B = [x*1000 for x in [0.06978759288787842, 0.2801102638244629, 0.6163508462905883, 1.0805705881118775, 1.7255639696121217,
    2.4955154180526735, 3.370841302871704, 4.517650356292725, 5.745762286186218, 7.038509984016418]]
    grah_milli_B = [x*1000 for x in [0.007873213291168213, 0.016229190826416016, 0.02476203203201294, 0.033674232959747315, 0.042335739135742186,
    0.050941919088363645, 0.05983875155448914, 0.06785545945167541, 0.07610330700874329, 0.08472154974937439]]
    mono_milli_B = [x*1000 for x in [0.009106448888778686, 0.01869452953338623, 0.02827052116394043, 0.038180583715438844, 0.04822150468826294,
    0.05885208964347839, 0.06831400275230408, 0.07893061757087708, 0.08912485957145691, 0.10061038613319397]]
    #graph showing average times against number of total points of data set B
    plt.plot(sizes, gift_milli_B, label="Giftwrap", linestyle='--')
    plt.plot(sizes, grah_milli_B, label="Grahamscan", linestyle='-.')
    plt.plot(sizes, mono_milli_B, label="Monotone chain")
    plt.xlabel("Number of input points")
    plt.ylabel("Time (ms)")
    plt.grid(color='b', linestyle='-', linewidth=.1)
    plt.legend(loc=2)
    plt.xticks(sizes)
    plt.show()

def graph_all(sizes):
    """Compares all algorithms over all data sets against input size
    """
    #The average times for the gift wrap algorithm in milli seconds
    gift_milli_B = [x*1000 for x in [0.06978759288787842, 0.2801102638244629, 0.6163508462905883, 1.0805705881118775, 1.7255639696121217,
    2.4955154180526735, 3.370841302871704, 4.517650356292725, 5.745762286186218, 7.038509984016418]]
    gift_milli_A = [x*1000 for x in [0.007160457372665405, 0.014404278993606568, 0.022417218685150148, 0.04061191082000733, 0.0503708827495575,
    0.059285746812820436, 0.06884238004684448, 0.09941885828971862, 0.10758164763450623, 0.12186283707618713]]
    #The average times for the Graham-scan algorithm in milli seconds
    grah_milli_B = [x*1000 for x in [0.007873213291168213, 0.016229190826416016, 0.02476203203201294, 0.033674232959747315, 0.042335739135742186,
    0.050941919088363645, 0.05983875155448914, 0.06785545945167541, 0.07610330700874329, 0.08472154974937439]]
    grah_milli_A = [x*1000 for x in [0.007544989585876465, 0.015188815593719483, 0.022944544553756715, 0.031321717500686644, 0.0395938789844513,
    0.0483058762550354, 0.05676034212112427, 0.06481868624687195, 0.07451822519302369, 0.08214217185974121]]
    #The average times for the Monotone chain algorithm in milli seconds
    mono_milli_B = [x*1000 for x in [0.009106448888778686, 0.01869452953338623, 0.02827052116394043, 0.038180583715438844, 0.04822150468826294,
    0.05885208964347839, 0.06831400275230408, 0.07893061757087708, 0.08912485957145691, 0.10061038613319397]]
    mono_milli_A = [x*1000 for x in [0.009165838956832886, 0.018670244216918944, 0.02833400249481201, 0.038080735206604, 0.04824641227722168,
    0.059222618341445925, 0.06878257393836976, 0.07932158946990966, 0.08988933444023132, 0.10021942853927612]]
    #graph showing average times against number of total points over all data sets
    plt.plot(sizes, gift_milli_A, label="Giftwrap - Set_A", linestyle='--', color="blue")
    #plt.plot(sizes, gift_milli_B, label="Giftwrap - Set_B")
    plt.plot(sizes, grah_milli_A, label="Graham-scan - Set_A", linestyle='--', color="red")
    plt.plot(sizes, grah_milli_B, label="Graham-scan - Set_B", color="red")
    plt.plot(sizes, mono_milli_A, label="Monotone chain - Set_A", linestyle='--', color="fuchsia")
    plt.plot(sizes, mono_milli_B, label="Monotone chain - Set_B", color="fuchsia")
    plt.xlabel("Number of input points")
    plt.ylabel("Time (ms)")
    plt.grid(color='b', linestyle='-', linewidth=.1)
    plt.legend(loc=2)
    plt.xticks(sizes)
    plt.show()

def graph_gift(sizes):
    """
        Outputs a grpah that shows the average time of the Gift wrap algorithm
        over both data sets
    """
    #The average times taken over data set A in milli seconds
    gift_milli_A = [x*1000 for x in [0.007160457372665405, 0.014404278993606568, 0.022417218685150148, 0.04061191082000733, 0.0503708827495575,
    0.059285746812820436, 0.06884238004684448, 0.09941885828971862, 0.10758164763450623, 0.12186283707618713]]
    #The average times taken over data set B in milli seconds
    gift_milli_B = [x*1000 for x in [0.06978759288787842, 0.2801102638244629, 0.6163508462905883, 1.0805705881118775, 1.7255639696121217,
    2.4955154180526735, 3.370841302871704, 4.517650356292725, 5.745762286186218, 7.038509984016418]]
    #Graph comparing the times taken in data set A and B for the gift wrap algorithm
    plt.plot(sizes, gift_milli_A, label="Data Set A", linestyle='-.')
    plt.plot(sizes, gift_milli_B, label="Data Set B", linestyle='--')
    plt.xlabel("Number of input points")
    plt.ylabel("Time (ms)")
    plt.grid(color='b', linestyle='-', linewidth=.1)
    plt.legend(loc=2)
    plt.xticks(sizes)
    plt.show()

def graph_graham(sizes):
    """
        Outputs a grpah that shows the average time of the Graham-scan algorithm
        over both data sets
    """
    #The average times taken over data set A
    grah_milli_A = [x*1000 for x in [0.007544989585876465, 0.015188815593719483, 0.022944544553756715, 0.031321717500686644, 0.0395938789844513,
    0.0483058762550354, 0.05676034212112427, 0.06481868624687195, 0.07451822519302369, 0.08214217185974121]]
    #The average times taken over data set B
    grah_milli_B = [x*1000 for x in [0.007873213291168213, 0.016229190826416016, 0.02476203203201294, 0.033674232959747315, 0.042335739135742186,
    0.050941919088363645, 0.05983875155448914, 0.06785545945167541, 0.07610330700874329, 0.08472154974937439]]
    #Graph comparing the times taken in data set A and B for the Graham-scan algorithm
    plt.plot(sizes, grah_milli_A, label="Data Set A", linestyle='-.')
    plt.plot(sizes, grah_milli_B, label="Data Set B", linestyle='--')
    plt.xlabel("Number of input points")
    plt.ylabel("Time (ms)")
    plt.grid(color='b', linestyle='-', linewidth=.1)
    plt.legend(loc=2)
    plt.xticks(sizes)
    plt.show()

def graph_mono(sizes):
    """
        Outputs a graph that shows the average time of the Monotone chain algorithm
        over both data sets
    """
    #The average times for the Monotone chan in milli seconds
    mono_milli_B = [x*1000 for x in [0.009106448888778686, 0.01869452953338623, 0.02827052116394043, 0.038180583715438844, 0.04822150468826294,
    0.05885208964347839, 0.06831400275230408, 0.07893061757087708, 0.08912485957145691, 0.10061038613319397]]
    mono_milli_A = [x*1000 for x in [0.009165838956832886, 0.018670244216918944, 0.02833400249481201, 0.038080735206604, 0.04824641227722168,
    0.059222618341445925, 0.06878257393836976, 0.07932158946990966, 0.08988933444023132, 0.10021942853927612]]
    #Graph comparing the times taken in data set A and B for the Monotone chain algorithm
    plt.plot(sizes, mono_milli_A, label="Data Set A", linestyle='--')
    plt.plot(sizes, mono_milli_B, label="Data Set B", linestyle='-.')
    plt.xlabel("Number of input points")
    plt.ylabel("Time (ms)")
    plt.grid(color='b', linestyle='-', linewidth=.1)
    plt.legend(loc=2)
    plt.xticks(sizes)
    plt.show()

def graph_all_hull(sizes_A, sizes_B):
    """Compares all algorithms over all data sets against convex hull size
    """
    #The average times for the gift wrap algorithm in milli seconds
    gift_milli_B = [x*1000 for x in [0.06978759288787842, 0.2801102638244629, 0.6163508462905883, 1.0805705881118775, 1.7255639696121217,
    2.4955154180526735, 3.370841302871704, 4.517650356292725, 5.745762286186218, 7.038509984016418]]
    gift_milli_A = [x*1000 for x in [0.007160457372665405, 0.014404278993606568, 0.022417218685150148, 0.04061191082000733, 0.0503708827495575,
    0.059285746812820436, 0.06884238004684448, 0.09941885828971862, 0.10758164763450623, 0.12186283707618713]]
    #The average times for the Graham-scan algorithm in milli seconds
    grah_milli_B = [x*1000 for x in [0.007873213291168213, 0.016229190826416016, 0.02476203203201294, 0.033674232959747315, 0.042335739135742186,
    0.050941919088363645, 0.05983875155448914, 0.06785545945167541, 0.07610330700874329, 0.08472154974937439]]
    grah_milli_A = [x*1000 for x in [0.007544989585876465, 0.015188815593719483, 0.022944544553756715, 0.031321717500686644, 0.0395938789844513,
    0.0483058762550354, 0.05676034212112427, 0.06481868624687195, 0.07451822519302369, 0.08214217185974121]]
    #The average times for the Monotone chain algorithm in milli seconds
    mono_milli_B = [x*1000 for x in [0.009106448888778686, 0.01869452953338623, 0.02827052116394043, 0.038180583715438844, 0.04822150468826294,
    0.05885208964347839, 0.06831400275230408, 0.07893061757087708, 0.08912485957145691, 0.10061038613319397]]
    mono_milli_A = [x*1000 for x in [0.009165838956832886, 0.018670244216918944, 0.02833400249481201, 0.038080735206604, 0.04824641227722168,
    0.059222618341445925, 0.06878257393836976, 0.07932158946990966, 0.08988933444023132, 0.10021942853927612]]
    fig, ax1 = plt.subplots()
    #graph showing average times against number of points in the convex hull over both data sets
    ax1.plot(sizes_A, gift_milli_A, label="Giftwrap - Set_A", color="fuchsia")
    #plt.plot(sizes, gift_milli_B, label="Giftwrap - Set_B")
    ax1.plot(sizes_A, grah_milli_A, label="Graham-scan - Set_A", linestyle='-.', color="fuchsia")
    ax1.plot(sizes_A, mono_milli_A, label="Monotone chain - Set_A", linestyle='--', color="fuchsia")
    ax1.set_xlabel("Number of convex hull points in data set A", color="fuchsia")
    ax1.set_ylabel("Time (ms)")
    ax1.set_xticks(sizes_A)
    ax1.tick_params(axis='x', labelcolor="fuchsia")
    #Adding second x axis
    ax2 = ax1.twiny()
    ax2.set_xlabel("Number of convex hull points in data set B", color="red")
    ax2.plot(sizes_B, mono_milli_B, label="Monotone chain - Set_B", linestyle='--', color="red")
    ax2.plot(sizes_B, grah_milli_B, label="Graham-scan - Set_B", linestyle='-.', color="red")
    ax2.set_xticks(sizes_B)
    ax2.tick_params(axis='x', labelcolor="red")
    plt.grid(color='b', linestyle='-', linewidth=.1)
    ax1.legend(loc=2)
    ax2.legend(loc=4)
    fig.tight_layout()
    plt.show()