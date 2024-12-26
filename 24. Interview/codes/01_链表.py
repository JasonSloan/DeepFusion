class Chain:
    def __init__(self, val, next):
        self.val = val
        self.next = next

if __name__ == '__main__':
    node3 = Chain(3, None)
    node2 = Chain(2, node3)
    node1 = Chain(1, node2)
    node0 = Chain(0, node1)
    head = node0
    for i in range(4):
        print(f"Node{i}节点的值为:{head.val}")
        head = head.next