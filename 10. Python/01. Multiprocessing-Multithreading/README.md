# 一. 参考资料

[链接](https://www.biaodianfu.com/python-multi-thread-and-multi-process.html)

# 二. 常用代码 

## 1. 获得当前进程ID

```python
import os
os.getpid()
```

## 2. 普通多线程

```python
from threading import Thread    

def test(name):
    print(name)

if __name__ == '__main__':
    t1 = Thread(target=test, args=('thread1',))
    t2 = Thread(target=test, args=('thread2',))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print('main')
```

## 3. 普通线程池

python由于GIL的存在, 多线程适用于IO操作比较多的函数

```python
import os
import concurrent.futures
import threading
import time

def process_image(image_path):
    file_name = os.path.basename(image_path)
    thread_id = threading.current_thread().ident
    print(f"Processing image: {file_name} in Thread ID: {thread_id}")
    time.sleep(0.1)

def process_images_in_parallel(image_paths, num_threads=4):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit each image processing task to the thread pool
        executor.map(process_image, image_paths)

if __name__ == "__main__":
    image_paths = [f"image_{i}" for i in range(10000)]
    num_threads = 4
    process_images_in_parallel(image_paths, num_threads)
```

## 3. 多线程之间共享进度条

```python
from multiprocessing.pool import ThreadPool
from itertools import repeat
import time
import random
from tqdm import tqdm

def verify_image_label(args):
    im_file, prefix = args
    time.sleep(random.uniform(0.1, 0.5))
    return im_file, 1


if __name__ == "__main__":
    im_files = [f'{i}.jpg' for i in range(1000)]
    nm = 0
    desc = 'Scanning'
    with ThreadPool(8) as pool:
        results = pool.imap(
            func=verify_image_label,
            iterable=zip(
                im_files,
                repeat("prefix"),
            ),
        )
        pbar = tqdm(results, desc=desc, total=len(im_files))
        for im_file, nm_f in pbar:
            nm += nm_f
            pbar.desc = f"{desc} {nm} images"
        pbar.close()
```

## 3. 普通多进程 

```python
from multiprocessing import Process
import os

def run_proc(name):
    print('Run child process %s (%s)...' % (name, os.getpid()))

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Process(target=run_proc, args=('test',))
    print('Child process will start.')
    p.start()
    p.join()
print('Child process end.')
```

## 4. 普通进程池 

```python
from multiprocessing import Pool

def test(i):
    print(i)

if __name__ == "__main__":
    lists = range(100)
    pool = Pool(8)
    print("多进程开始执行")
    pool.map(test, lists)
    pool.close()
    pool.join()
```

## 5. 向进程池中添加任务(全部添加结束后执行) 

向进程池中添加任务, 所有任务全部添加完才执行, 进程池中进程数为8, 也就是开始执行后每次最多同时执行8个任务

```python
from multiprocessing import Pool

def test(i):
    print(i)

if __name__ == "__main__":
    pool = Pool(8)
    for i in range(100):
        '''
        For循环中执行步骤：
        （1）循环遍历，将100个子进程添加到进程池（相对父进程会阻塞）
        （2）每次执行8个子进程，等一个子进程执行完后，立马启动新的子进程。（相对父进程不阻塞）
        apply_async为异步进程池写法。异步指的是启动子进程的过程，与父进程本身的执行（print）是异步的，而For循环中往进程池添加子进程的过程，与父进程本身的执行却是同步的。
        '''
        # 只是添加进程并不执行，注意，如果向test传参的时候需要传一个实例对象，那么该实例对象要在for循环内实例化，以保证每个进程都拥有一个独立的实例对象，如果在for循环外实例化一个，然后所有进程共享，会出现错误
        pool.apply_async(test, args=(i,))  # 维持执行的进程总数为8，当一个进程执行完后启动一个新进程.
    # 关闭进程池
    pool.close()
    print("多进程开始执行")
    # 等待子进程结束后再继续往下运行，通常用于进程间的同步
    pool.join()
    print("多进程结束执行")
```

## 6. 向进程池中添加任务(边添加边执行) 

向进程池中添加任务, 只要有任务就执行, 进程池中进程数为8, 也就是最多同时执行8个任务

```python
from multiprocessing import Pool

def test(i):
    print(i)

if __name__ == "__main__":
    pool = Pool(8)
    for i in range(100):
        '''
            实际测试发现，for循环内部执行步骤：
            （1）遍历100个可迭代对象，往进程池放一个子进程
            （2）执行这个子进程，等子进程执行完毕，再往进程池放一个子进程，再执行。（同时只执行一个子进程）
            for循环执行完毕，再执行print函数。
        '''
        # 注意，如果向test传参的时候需要传一个实例对象，那么该实例对象要在for循环内实例化，以保证每个进程都拥有一个独立的实例对象，如果在for循环外实例化一个，然后所有进程共享，会出现错误
        pool.apply(test, args=(i,))  # 维持执行的进程总数为8，当一个进程执行完后启动一个新进程.
    print("多进程结束执行")
    pool.close()
    pool.join()
```

## 7. 进程池间数据共享

```python
from multiprocessing import Pool, Value

def worker(index):
    for _ in range(5):
        with shared_value.get_lock():
            shared_value.value += 1
        print(f"Worker {index}: {shared_value.value}")

if __name__ == "__main__":
    # 'i' 代表属于整数integer
    shared_value = Value('i', 0)  
    pool_size = 3
    pool = Pool(pool_size)
    results = [pool.apply_async(worker, args=(i,)) for i in range(pool_size)]
    pool.close()
    pool.join()
    print(f"Main: {shared_value.value}")
```

**注意: 进程池间数据共享, 传参时每个参数一定要是独立的**

```python
# 错误的写法
from multiprocessing import Pool, Value

class ID():
    def __init__(self):
        self.id = 0
        
def worker(id_cls):
    print(f"Worker {id_cls.id}")

if __name__ == "__main__":
    pool_size = 3
    pool = Pool(pool_size)
    id_cls = ID()
    for i in range(pool_size):
        # 因为pool.apply_async是对id_cls进行引用, 所以worker调用的结果都是同一个值
        # 正确写法应该是将id_cls在循环内部实例化
        id_cls.id = i
        pool.apply_async(func=worker, args=(id_cls,))
    pool.close()
    pool.join()
# 三次打印结果: Worker 2 Worker 2 Worker 2
```

```python
# 正确的写法
from multiprocessing import Pool, Value

class ID():
    def __init__(self):
        self.id = 0
        
def worker(id_cls):
    print(f"Worker {id_cls.id}")

if __name__ == "__main__":
    pool_size = 3
    pool = Pool(pool_size)
    for i in range(pool_size):
        # 每一个进程都应该独占自己的ID类
        id_cls = ID()
        id_cls.id = i
        pool.apply_async(func=worker, args=(id_cls,))
    pool.close()
    pool.join()
# 三次打印结果: Worker 0 Worker 1 Worker 2
```

## 8. JoinableQueue实现多进程之间的通信

多进程间的通信(JoinableQueue)
task_done()：消费者使用此方法发出信号，表示q.get()的返回项目已经被处理。如果调用此方法的次数大于从队列中删除项目的数量，将引发ValueError异常
join():生产者调用此方法进行阻塞，直到队列中所有的项目均被处理。阻塞将持续到队列中的每个项目均调用q.task_done（）方法为止

```python
from multiprocessing import Process, JoinableQueue
import time, random

def consumer(q):
    while True:
        res = q.get()
        print('消费者拿到了 %s' % res)
        q.task_done()

def producer(seq, q):
    for item in seq:
        time.sleep(random.randrange(1,2))
        q.put(item)
        print('生产者做好了 %s' % item)
    q.join()

if __name__ == "__main__":
    q = JoinableQueue()
    seq = ('产品%s' % i for i in range(5))
    p = Process(target=consumer, args=(q,))
    p.daemon = True  # 设置为守护进程，在主线程停止时p也停止，但是不用担心，producer内调用q.join保证了consumer已经处理完队列中的所有元素
    p.start()
    producer(seq, q)
    print('主线程')
```

## 9. 进程间数据共享(不常用)

进程间的数据共享(multiprocessing.Queue)(基本不用, 因为进程间本来就是资源独立的)

```python
from multiprocessing import Process, Queue
import os, time, random

def write(q):
    print('Process to write: %s' % os.getpid())
    for value in ['A', 'B', 'C']:
        print('Put %s to queue...' % value)
        q.put(value)
        time.sleep(random.random())

def read(q):
    print('Process to read: %s' % os.getpid())
    while True:
        value = q.get(True)
        print('Get %s from queue.' % value)

if __name__ == "__main__":
    q = Queue()
    pw = Process(target=write, args=(q,))
    pr = Process(target=read, args=(q,))
    pw.start()
    pr.start()
    pw.join()  # 等待pw结束
    pr.terminate()  # pr进程里是死循环，无法等待其结束，只能强行终止
```

