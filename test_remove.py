from collections import Counter
import numpy as np
max_count = 10


def remove_bad_matches(list_of_matches1, list_of_matches2):
    temp_list = Counter(str(e) for e in list_of_matches1)
    bad_list = []
    for i in temp_list: 
        if temp_list[i] > max_count:
            bad_list.append(i)
    counter = len(list_of_matches1) - 1

    #nump delete creates new array, so should only call once
    indexes_to_delete = []
    while(counter >= 0):
        if str(list_of_matches1[counter]) in bad_list:
            indexes_to_delete.append(counter)
        counter -= 1

    return np.delete(list_of_matches1, indexes_to_delete, axis=0), np.delete(list_of_matches2, indexes_to_delete, axis=0)

a = np.array([[1,2,3],[3,4,5],[1,2,3]])
b = np.array([[1,1,1],[2,2,2],[3,3,3]])
remove_bad_matches(a,b)
print a
print b