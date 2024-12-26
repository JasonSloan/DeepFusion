import numpy as np


def sigmoid(x):
    return 1. / (1. + np.exp(-x))

class BP:
   
    def __init__(self, in_dim, hidden_dim) -> None:
        self.w1 = np.random.randn(in_dim, hidden_dim)
        self.b1 = np.random.randn(hidden_dim)
        self.w2 = np.random.randn(hidden_dim, 1)
        self.b2 = np.random.randn(1)
    
    def forward(self, x, y):
        # x : [bs, in_dim]
        z1 = x@self.w1 + self.b1        # z1 : [bs, hidden_dim]
        a1 = sigmoid(z1)                # a1 : [bs, hidden_dim]
        z2 = a1@self.w2 + self.b2       # z2 : [bs, 1]

        # y : [bs, 1]
        loss = 0.5 * np.square(z2 - y).mean(0)
        print(f"Loss:{loss.item():.2f}")
        
        # use BGD
        loss_grad_z2 = (z2 - y)             # loss_grad_z2 : [bs, 1]
        z2_grad_w2 = a1                     # z2_grad_w2 : [bs, hidden_dim]
        loss_grad_w2 = z2_grad_w2.T @ loss_grad_z2  # loss_grad_w2 : [hidden_dim, 1]
        loss_grad_b2 = loss_grad_z2.sum(0)
        
        z2_grad_a1 = self.w2                    # z2_grad_a1 : [hidden_dim, 1]
        a1_grad_z1 = a1 * (1 - a1)              # a1_grad_z1 : [bs, hidden_dim] 
        z1_grad_w1 = x                          # z1_grad_w1 : [bs, in_dim]
        loss_grad_w1 = z1_grad_w1.T @ (loss_grad_z2 @ z2_grad_a1.T * a1_grad_z1)
        loss_grad_b1 = (loss_grad_z2 @ z2_grad_a1.T * a1_grad_z1).sum(0)
        
        self.grads = [loss_grad_w2, loss_grad_b2, loss_grad_w1, loss_grad_b1]
        return z2
    
    def backward(self, lr=0.001):
        loss_grad_w2, loss_grad_b2, loss_grad_w1, loss_grad_b1 = self.grads
        self.w2 -= lr * loss_grad_w2
        self.b2 -= lr * loss_grad_b2
        self.w1 -= lr * loss_grad_w1
        self.b1 -= lr * loss_grad_b1


if __name__ == "__main__":
    x = np.random.randn(4, 3)
    # provided its a regression problem
    y = x[:, 0] * 2 + x[:, 1] * 3 + x[:, 2] * 4 + np.random.rand()
    y = y[..., None]
    bp = BP(3, 128)
    for i in range(10000):
        bp.forward(x, y)
        bp.backward()
    print(y)
    print(bp.forward(x, y))
        
    
        
        
         
        