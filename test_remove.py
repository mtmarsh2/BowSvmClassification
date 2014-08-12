from collections import Counter
max_count = 2


def remove_bad_matches(list_of_matches1, list_of_matches2):
    temp_list = Counter(str(e) for e in list_of_matches1)
    bad_list = []
    for i in temp_list: 
        if temp_list[i] > 1:
            bad_list.append(i)
    counter = len(list_of_matches1) - 1
    while(counter >= 0):
        if str(list_of_matches1[counter]) in bad_list:
            del list_of_matches1[counter]
            del list_of_matches2[counter]
        counter -= 1
