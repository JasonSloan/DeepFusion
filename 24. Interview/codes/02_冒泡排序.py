# 时间复杂度O(n^2)
import queue


def bubble(l):
    length = len(l)
    for i in range(length):
        for j in range(length - i - 1):
            if l[j] > l[j + 1]:
                tmp = l[j]
                l[j] = l[j + 1]
                l[j + 1] = tmp


if __name__ == '__main__':
    l = [3, 56, 7, 154, 723, 7, 8]
    bubble(l)
    print(l)


from queue import LifoQueue
stack = LifoQueue()
stack.put()
