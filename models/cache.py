import torch
class Node:
    def __init__(self, key=0, val=0, freq=0,embed=None):
        self.key = key
        self.val = val
        self.embed = embed
        self.freq = freq
        self.prev = None
        self.next = None

class LFUCacheE:
    def __init__(self, capacity: int, ways:int):
        self.cache = {}  # 存储缓存的内容
        self.freq = {}  # 存储每个频次对应的双向链表
        self.ncap = capacity
        self.size = 0
        # self.time = 0
        self.minFreq = 0  # minFreq存储当前最小频次
        self.ways = ways

    def get(self, embed) :
        # set hash key
        key = int(round(embed.sum().tolist(),8)*1e8)
        
        if key not in self.cache:
            return

        node = self.cache[key]
        self.incFreq(node)
        # return node.val
        return

    def get_all(self):
        label2embed = {i:torch.tensor([]) for i in range(self.ways)}
        for key in self.cache:
            node = self.cache[key]
            label2embed[node.val] = torch.cat((label2embed[node.val], node.embed.unsqueeze(0)), dim=0)

        return label2embed

    def put(self, embed, value) -> None:
        '''
        key: query's label
        value: query's embedding
        '''
        if self.ncap <= 0:
            return

        # set hash key
        key = int(round(embed.sum().tolist(),8)*1e8)

        if key in self.cache:
            node = self.cache[key]
            # node.val = value
            self.incFreq(node)
        else:
            if self.size >= self.ncap:
                # 通过 minFreq 拿到 freq_table[minFreq] 链表的末尾节点
                node = self.freq[self.minFreq].removeLast()
                del self.cache[node.key]
                self.size -= 1

            x = Node(key, value, 1, embed)
            self.cache[key] = x
            if 1 not in self.freq:
                self.freq[1] = DoubleList()
            self.freq[1].addFirst(x)
            self.minFreq = 1
            self.size += 1

        # self.time += 1

    def incFreq(self, node: Node) -> None:
        _freq = node.freq
        self.freq[_freq].remove(node)

        if self.minFreq == _freq and self.freq[_freq].isEmpty():
            self.minFreq += 1
            del self.freq[_freq]

        node.freq += 1
        if node.freq not in self.freq:
            self.freq[node.freq] = DoubleList()
        self.freq[node.freq].addFirst(node)


class DoubleList:
    def __init__(self):
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def addFirst(self, node: Node) -> None:
        node.next = self.head.next
        node.prev = self.head

        self.head.next.prev = node
        self.head.next = node

    def remove(self, node: Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev

        node.next = None
        node.prev = None

    def removeLast(self) -> Node:
        if self.isEmpty():
            return None

        last = self.tail.prev
        self.remove(last)

        return last

    def isEmpty(self) -> bool:
        return self.head.next == self.tail



class LFUCache:

    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.freq = {}
        self.time = 0
        self.max_freq = 1
        self.min_freq = 1

    def get(self, key):
        if key not in self.cache:
            return -1

        # 更新频率
        freq = self.freq[key]
        self.freq[key] = freq + 1
        if freq == self.max_freq:
            self.max_freq = self.freq[key]

        # 更新最小频率
        if freq == self.min_freq:
            self.min_freq = min(self.freq.values())

        return self.cache[key]

    def put(self, key):
        if self.capacity == 0:
            return

        # 如果 key 已经存在，更新值并更新频率
        if key in self.cache:
            freq = self.freq[key]
            self.freq[key] = freq + 1
            if freq == self.max_freq:
                self.max_freq += 1

            self.cache[key] = self.time
            self.time += 1

            # 更新最小频率
            if freq == self.min_freq:
                self.min_freq = min(self.freq.values())
            return self.freq[key]

        # 如果缓存已满，删除最久最少使用的缓存
        if len(self.cache) == self.capacity:
            min_freq_keys = [
                k for k, v in self.freq.items() if v == self.min_freq
            ]
            oldest_key = min(min_freq_keys, key=lambda k: self.cache[k])
            del self.cache[oldest_key]
            del self.freq[oldest_key]
            for key in self.freq.keys():
                self.freq[key] = self.freq[key] - 0.5 if self.freq[
                    key] > 1 else self.freq[key]

        # 添加新缓存
        self.cache[key] = self.time
        self.freq[key] = 1
        self.min_freq = 1
        self.time += 1
        return self.freq[key]
