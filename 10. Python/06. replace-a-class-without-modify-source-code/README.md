如果有这么个需求，有一个类，他的成员方法中使用到了类的成员属性，在不改变原代码的情况下，如何替换掉这个方法？

如下：

```python

class Trainer:
    def __init__(self, model="resnet18"):
        self.model = model

    def train(self):
        print(f"Training v1 wiht model {self.model}")


def train_v2(self: Trainer, extra_arg=False):
    print(f"Training v2 wiht model {self.model}")
    if extra_arg:
        print("Do sth with extra_arg")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
    # 由于trainner.train方法中使用到了self.model, self.model是Trainer类的属性
    # 重写trainer类的train方法，也需要使用到trainer实例的self.model属性
    # 如果想使用trainer.train = train_v2来替换trainer的train方法是不可行的，因为这样无法在train_v2中使用到self.model了
    # 所以不改变任何原代码的情况下，需要使用trainer.__setattr__方法来设置trainer实例的属性
    trainer.__setattr__("train", train_v2.__get__(trainer))
    trainer.train()
```

