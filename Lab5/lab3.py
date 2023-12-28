class ArithmeticEncoder:
    def __init__(self):
        self.low = 0.0
        self.high = 1.0
        self.range = 1.0

    def encode(self, probabilities, message):
        for symbol in message:
            low_range = self.low + self.range * sum(probabilities[:symbol])
            high_range = self.low + self.range * sum(probabilities[:symbol + 1])

            self.low = low_range
            self.high = high_range
            self.range = self.high - self.low

        return (self.low + self.high) / 2

# 信息和概率分布
message = "bcadc"
probabilities = {'a': 0.2, 'b': 0.3, 'c': 0.4, 'd': 0.1}

# 将概率转换为累积概率
sorted_probabilities = [probabilities[symbol] for symbol in sorted(probabilities)]

# 创建并使用算术编码器
encoder = ArithmeticEncoder()
encoded_result = encoder.encode(sorted_probabilities, [ord(symbol) - ord('a') for symbol in message])

print("编码结果区间：", encoded_result)
