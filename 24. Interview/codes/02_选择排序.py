

def choose(l):
    for i in range(len(l)):
        min_num_idx = i
        for j in range(i, len(l)):
            if l[j] < l[min_num_idx]:
                min_num_idx = j
        l[i], l[min_num_idx] = l[min_num_idx], l[i]


if __name__ == '__main__':
    l = [3, 56, 7, 154, 723, 7, 8, 4, 2, 1, 8, 6, 5, 4, 3]
    choose(l)
    print(l)
