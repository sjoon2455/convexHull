import convexhull as convex
import convexhull_time as con_time
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
    sizes = [3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000]
    """
        These filenames may need to be altered dependent on their location
    """
    filenames_a = ["Set_A/A_3000.dat", "Set_A/A_6000.dat", "Set_A/A_9000.dat", "Set_A/A_12000.dat", "Set_A/A_15000.dat",
    "Set_A/A_18000.dat", "Set_A/A_21000.dat", "Set_A/A_24000.dat", "Set_A/A_27000.dat", "Set_A/A_30000.dat"]
    filenames_b = ["Set_B/B_3000.dat", "Set_B/B_6000.dat", "Set_B/B_9000.dat", "Set_B/B_12000.dat", "Set_B/B_15000.dat",
    "Set_B/B_18000.dat", "Set_B/B_21000.dat", "Set_B/B_24000.dat", "Set_B/B_27000.dat", "Set_B/B_30000.dat"]
    gift = []
    graham = []
    mono = []
    if flag:
        #Testing data set A
        for index, filename in enumerate(filenames_a):
            gift_times = []
            grah_times = []
            mono_times = []
            #Getting points from file
            listPts = con_time.readDataPts(filename, sizes[index])
            print(filename, "\n----------")
            #Getting points from file
            for i in range(amount):
                #calling each algorithm
                gft_hull, gift_time = con_time.giftwrap(listPts[:])
                grs_hull, grah_time = con_time.grahamscan(listPts[:])
                mon_hull, mono_time = con_time.monotone_chain(listPts[:])
                #Adding time taken to list of times
                gift_times.append(gift_time)
                grah_times.append(grah_time)
                mono_times.append(mono_time)
            #Getting average time
            gift.append(sum(gift_times)/len(gift_times))
            graham.append(sum(grah_times)/len(grah_times))
            mono.append(sum(mono_times)/len(mono_times))
    else:
        #Testing data set B
        for index, filename in enumerate(filenames_b):
            gift_times = []
            grah_times = []
            mono_times = []
            #Getting points from file
            listPts = con_time.readDataPts(filename, sizes[index])
            print(filename, "\n----------")
            #Getting points from file
            for i in range(amount):
                #calling each algorithm
                gft_hull, gift_time = con_time.giftwrap(listPts[:])
                grs_hull, grah_time = con_time.grahamscan(listPts[:])
                mon_hull, mono_time = con_time.monotone_chain(listPts[:])
                #Adding time taken to list of times
                gift_times.append(gift_time)
                grah_times.append(grah_time)
                mono_times.append(mono_time)
            #Getting average time
            gift.append(sum(gift_times)/len(gift_times))
            graham.append(sum(grah_times)/len(grah_times))
            mono.append(sum(mono_times)/len(mono_times))
    #Outputting results
    print("gift:\n", gift)
    print("graham:\n", graham)
    print("mono:\n", mono)

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
    #Looping over each file in set A
    for index, filename in enumerate(filenames_a):
        #Testing data set A
        print(filename)
        #Getting points
        listPts = convex.readDataPts(filename, sizes[index])
        #Calling algorithms
        gft_hull = convex.giftwrap(listPts[:])
        grs_hull = convex.grahamscan(listPts[:])
        mono_hull = convex.monotone_chain(listPts[:])
        #Getting result of the algorithm
        expected = outputs_A[index]
        gft = "Fail"
        grs = "Fail"
        mono = "Fail"
        if gft_hull == expected: gft = "Pass"
        if grs_hull == expected: grs = "Pass"
        if sorted(mono_hull) == sorted(expected): mono = "Pass"
        #Outputting results of the test
        print("Gifftwrap: " + gft)
        print("Grahamscan: " + grs)
        print("Monotone chain: " + mono)
        print("---------------")
    print("\nSet B")
    print("---------------")
    #Looping over each file in set B
    for index, filename in enumerate(filenames_b):
        #Testing data set B
        print(filename)
        #Getting points
        listPts = convex.readDataPts(filename, sizes[index])
        #Calling algorithms
        gft_hull = convex.giftwrap(listPts[:])
        grs_hull = convex.grahamscan(listPts[:])
        mono_hull = convex.monotone_chain(listPts[:])
        #Getting result of the algorithm
        expected = outputs_B[index]
        gft = "Fail"
        grs = "Fail"
        mono = "Fail"
        if gft_hull == expected: gft = "Pass"
        if grs_hull == expected: grs = "Pass"
        if sorted(mono_hull) == sorted(expected): mono = "Pass"
        #Outputting results of the test
        print("Gifftwrap: " + gft)
        print("Grahamscan: " + grs)
        print("Monotone chain: " + mono)
        print("---------------")