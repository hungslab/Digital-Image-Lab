import heapq
from collections import defaultdict
import math

def huffman_coding(image_data):
    # 构建哈夫曼树
    def build_tree(freq):
        heap = [[weight, [symbol, '']] for symbol, weight in freq.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        return heap[0][1:]

    frequencies = defaultdict(float)
    for i, prob in enumerate(image_data):
        frequencies[f'y{i+1}'] = prob

    # 构建哈夫曼编码
    huffman_tree = build_tree(frequencies)
    huffman_codes = {symbol: code for symbol, code in huffman_tree}

    # 计算平均码长
    average_code_length = sum([prob * len(huffman_codes[symbol]) for symbol, prob in frequencies.items()])

    # 计算编码效率
    entropy = sum([-prob * (prob and math.log2(prob)) for prob in frequencies.values()])
    coding_efficiency = entropy / average_code_length

    # 计算压缩比
    original_bits = 3  # 需要3个比特量化
    encoded_bits = sum([prob * len(huffman_codes[symbol]) for symbol, prob in frequencies.items()])

    compression_ratio = original_bits / encoded_bits

    # 计算冗余度
    redundancy = 1 - coding_efficiency

    return huffman_codes, average_code_length, coding_efficiency, compression_ratio, redundancy

# 给定灰度级和概率
image_grayscale = [0.40, 0.18, 0.10, 0.10, 0.07, 0.06, 0.05, 0.04]

huffman_result = huffman_coding(image_grayscale)
print(f"Huffman 编码表：", huffman_result[0])
print(f"平均码长：{huffman_result[1]:.2f}")
print(f"编码效率：{huffman_result[2]:.2f}")
print(f"压缩比：{huffman_result[3]:.2f}")
print(f"冗余度：{huffman_result[4] * 100:.2f}%")