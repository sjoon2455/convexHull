import convexhull as convex
import convexhull_time as con_time
import symmetric as sym
import binarytree as btr
import time
from outputs import *
################################
################################
"""
    Lists holding filenames may need to be altered to perform tests
    if the data files are not in the same location
"""
################################
################################


def average_tests(amount, flag):
    """
        Runs tests which give the aveage time for each algorithm over all data sets
    """
    sizes = [3000, 6000]
    # sizes = [3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000]
    """
        These filenames may need to be altered dependent on their location
    """
    filenames_a = ["Set_A/A_3000.dat", "Set_A/A_6000.dat", "Set_A/A_9000.dat", "Set_A/A_12000.dat", "Set_A/A_15000.dat",
                   "Set_A/A_18000.dat", "Set_A/A_21000.dat", "Set_A/A_24000.dat", "Set_A/A_27000.dat", "Set_A/A_30000.dat"]
    # filenames_b = ["Set_B/B_3000.dat", "Set_B/B_6000.dat", "Set_B/B_9000.dat", "Set_B/B_12000.dat", "Set_B/B_15000.dat",
    #                "Set_B/B_18000.dat", "Set_B/B_21000.dat", "Set_B/B_24000.dat", "Set_B/B_27000.dat", "Set_B/B_30000.dat"]
    filenames_b = ["Set_B/B_3000.dat"]

    gift = []
    graham = []
    mono = []
    incrmt = []
    chan = []
    quick = []
    ks = []
    dnq = []
    melk = []
    bt = []
    symm = []
    if flag:
        # Testing data set A
        for index, filename in enumerate(filenames_a):
            gift_times = []
            grah_times = []
            mono_times = []
            incrmt_times = []
            chan_times = []
            quick_times = []
            ks_times = []
            dnq_times = []
            melk_times = []
            bt_times = []
            symm_times = []
            # Getting points from file
            listPts = con_time.readDataPts(filename, sizes[index])
            print(filename, "\n----------")
            # Getting points from file
            for i in range(amount):
                # calling each algorithm
                gft_hull, gift_time = con_time.giftwrap(listPts[:])
                grs_hull, grah_time = con_time.grahamscan(listPts[:])
                mon_hull, mono_time = con_time.monotone_chain(listPts[:])
                incrmt_hull, incrmt_time = convex.incremental(listPts[:])
                chan_hull, chan_time = convex.Chan(listPts[:])
                quick_hull, quick_time = convex.quickHull(listPts[:])
                ks_hull, ks_time = convex.kirk_seidel(listPts[:])
                dnq_hull, dnq_time = convex.convex_hull_recursive(listPts[:])
                melk_hull, melk_time = convex.convex_hull_melkman(listPts[:])
                bt_hull, bt_time = btr.binarytree(listPts[:])
                symm_hull, symm_time = sym.symmetric(listPts[:])
                # Adding time taken to list of times
                gift_times.append(gift_time)
                grah_times.append(grah_time)
                mono_times.append(mono_time)
                incrmt_times.append(incrmt_time)
                chan_times.append(chan_time)
                quick_times.append(quick_time)
                ks_times.append(ks_time)
                dnq_times.append(dnq_time)
                melk_times.append(melk_time)
                bt_times.append(bt_time)
                symm_times.append(symm_time)

            # Getting average time
            gift.append(sum(gift_times)/len(gift_times))
            graham.append(sum(grah_times)/len(grah_times))
            mono.append(sum(mono_times)/len(mono_times))
            incrmt.append(sum(incrmt_times)/len(incrmt_times))
            chan.append(sum(chan_times)/len(chan_times))
            quick.append(sum(quick_times)/len(quick_times))
            ks.append(sum(ks_times)/len(ks_times))
            dnq.append(sum(dnq_times)/len(dnq_times))
            melk.append(sum(melk_times)/len(melk_times))
            bt.append(sum(bt_times)/len(bt_times))
            symm.append(sum(symm_times)/len(symm_times))
    else:
        # Testing data set B
        for index, filename in enumerate(filenames_b):
            gift_times = []
            grah_times = []
            mono_times = []
            incrmt_times = []
            chan_times = []
            quick_times = []
            ks_times = []
            dnq_times = []
            melk_times = []
            bt_times = []
            symm_times = []
            # Getting points from file
            listPts = con_time.readDataPts(filename, sizes[index])
            print(filename, "\n----------")
            # Getting points from file
            for i in range(amount):
                # calling each algorithm
                gft_hull, gift_time = con_time.giftwrap(listPts[:])
                grs_hull, grah_time = con_time.grahamscan(listPts[:])
                mon_hull, mono_time = con_time.monotone_chain(listPts[:])
                incrmt_hull, incrmt_time = convex.incremental(listPts[:])
                chan_hull, chan_time = convex.Chan(listPts[:])
                quick_hull, quick_time = convex.quickHull(listPts[:])
                ks_hull, ks_time = convex.kirk_seidel(listPts[:])
                dnq_hull, dnq_time = convex.convex_hull_recursive(listPts[:])
                melk_hull, melk_time = convex.convex_hull_melkman(listPts[:])
                bt_hull, bt_time = btr.binarytree(listPts[:])
                symm_hull, symm_time = sym.symmetric(listPts[:])
                # Adding time taken to list of times
                gift_times.append(gift_time)
                grah_times.append(grah_time)
                mono_times.append(mono_time)
                incrmt_times.append(incrmt_time)
                chan_times.append(chan_time)
                quick_times.append(quick_time)
                ks_times.append(ks_time)
                dnq_times.append(dnq_time)
                melk_times.append(melk_time)
                bt_times.append(bt_time)
                symm_times.append(symm_time)
            # Getting average time
            gift.append(sum(gift_times)/len(gift_times))
            graham.append(sum(grah_times)/len(grah_times))
            mono.append(sum(mono_times)/len(mono_times))
            incrmt.append(sum(incrmt_times)/len(incrmt_times))
            chan.append(sum(chan_times)/len(chan_times))
            quick.append(sum(quick_times)/len(quick_times))
            ks.append(sum(ks_times)/len(ks_times))
            dnq.append(sum(dnq_times)/len(dnq_times))
            melk.append(sum(melk_times)/len(melk_times))
            bt.append(sum(bt_times)/len(bt_times))
            symm.append(sum(symm_times)/len(symm_times))
    # Outputting results
    print("gift:\n", gift)
    print("graham:\n", graham)
    print("mono:\n", mono)
    print("incrmt:\n", incrmt)
    print("chan:\n", chan)
    print("quick:\n", quick)
    print("kirk-seidel:\n", ks)
    print("divide-n-conquer:\n", dnq)
    print("melkman:\n", melk)
    print("bt:\n", bt)
    print("symm:\n", symm)


def output_tests():
    """
        Runs tests that validate all algrithms
    """
    sizes = [3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000]
    """
        These filenames may need to be altered dependent on their location
    """
    filenames_a = ["Set_A/A_3000.dat", "Set_A/A_6000.dat", "Set_A/A_9000.dat", "Set_A/A_12000.dat", "Set_A/A_15000.dat",
                   "Set_A/A_18000.dat", "Set_A/A_21000.dat", "Set_A/A_24000.dat", "Set_A/A_27000.dat", "Set_A/A_30000.dat"]
    filenames_b = ["Set_B/B_3000.dat", "Set_B/B_6000.dat", "Set_B/B_9000.dat", "Set_B/B_12000.dat", "Set_B/B_15000.dat",
                   "Set_B/B_18000.dat", "Set_B/B_21000.dat", "Set_B/B_24000.dat", "Set_B/B_27000.dat", "Set_B/B_30000.dat"]
    print("Set A")
    print("---------------")
    # Looping over each file in set A
    for index, filename in enumerate(filenames_a):
        # Testing data set A
        print(filename)
        # Getting points
        listPts = convex.readDataPts(filename, sizes[index])
        # Calling algorithms
        gft_hull = convex.giftwrap(listPts[:])
        grs_hull = convex.grahamscan(listPts[:])
        mono_hull = convex.monotone_chain(listPts[:])
        # Getting result of the algorithm
        expected = outputs_A[index]
        gft = "Fail"
        grs = "Fail"
        mono = "Fail"
        if gft_hull == expected:
            gft = "Pass"
        if grs_hull == expected:
            grs = "Pass"
        if sorted(mono_hull) == sorted(expected):
            mono = "Pass"
        # Outputting results of the test
        print("Gifftwrap: " + gft)
        print("Grahamscan: " + grs)
        print("Monotone chain: " + mono)
        print("---------------")
    print("\nSet B")
    print("---------------")
    # Looping over each file in set B
    for index, filename in enumerate(filenames_b):
        # Testing data set B
        print(filename)
        # Getting points
        listPts = convex.readDataPts(filename, sizes[index])
        # Calling algorithms
        gft_hull = convex.giftwrap(listPts[:])
        grs_hull = convex.grahamscan(listPts[:])
        mono_hull = convex.monotone_chain(listPts[:])
        # Getting result of the algorithm
        expected = outputs_B[index]
        gft = "Fail"
        grs = "Fail"
        mono = "Fail"
        if gft_hull == expected:
            gft = "Pass"
        if grs_hull == expected:
            grs = "Pass"
        if sorted(mono_hull) == sorted(expected):
            mono = "Pass"
        # Outputting results of the test
        print("Gifftwrap: " + gft)
        print("Grahamscan: " + grs)
        print("Monotone chain: " + mono)
        print("---------------")
