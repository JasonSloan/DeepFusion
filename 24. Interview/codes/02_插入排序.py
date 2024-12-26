def insert(l):
    for i in range(1, len(l)):
        sorted_part_current_idx = i - 1
        unsorted_part_current_value = l[i]
        gap_idx = i
        while l[sorted_part_current_idx] > unsorted_part_current_value and sorted_part_current_idx >= 0:
            l[gap_idx] = l[sorted_part_current_idx]
            sorted_part_current_idx -= 1
            gap_idx-=1
        l[gap_idx] = unsorted_part_current_value







if __name__ == '__main__':
    l = [3, 56, 7, 154, 723, 7, 8, 4, 2, 1, 8, 6, 5, 4, 3]
    insert(l)
    print(l)
