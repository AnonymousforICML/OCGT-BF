import numpy as np
from nda.optimizers import compressor as cop
import math

# import numpy as np

def identity(x, *args, **kwargs):
    return x


# top_a
def top(x, a):
    dim = x.shape[0]
    if a == 0:
        return 0
    if a >= dim:
        return x
    index_array = np.argpartition(x, kth=a, axis=0)[a:]
    np.put_along_axis(x, index_array, 0, axis=0)
    return x


# x = np.random.randint(0, 100, 24).reshape(6, 4)
# x
# top(x, 2)

# Random_a compressor, keep a values
def random(x, a):
    dim = x.shape[0]
    if a == 0:
        return 0
    if a == dim:
        return x
    if x.ndim == 2:
        for i in range(x.shape[1]):
            zero_mask = np.random.choice(dim, dim - a, replace=False)
            x[zero_mask, i] = 0
    else:
        zero_mask = np.random.choice(dim, dim - a, replace=False)
        x[zero_mask] = 0
    return x


# gsgd_b
def gsgd(x, b):
    norm = np.linalg.norm(x, axis=0)
    return norm / (2 ** (b - 1)) * np.sign(x) * np.floor(
        (2 ** (b - 1)) / norm * np.abs(x) + np.random.uniform(0, 1, x.shape)
    )


# random quantization 2-norm with level s
def random_quantization(x, s):
    # 获取输入向量 x 的维度
    dim = x.shape[0]
    # 计算输入向量 x 的二范数（欧几里得范数）
    xnorm = np.linalg.norm(x)

    # 检查量化级别 s 或范数 xnorm 是否为零
    if s == 0 or xnorm == 0:
        # 如果 s 或 xnorm 有一个为零，则返回相同维度的零填充数组
        return np.zeros(dim, dtype=int)

    # 为向量 x 中的每个元素生成随机噪声
    noise = np.random.uniform(0, 1, size=x.shape)

    # 执行随机量化，使用公式：floor(s * |x| / xnorm + noise)

    rounded = np.floor(s * np.abs(x) / xnorm + noise)

    compressed = (xnorm / s) * np.sign(x) * rounded
    return compressed


# random quantization 2-norm with level s
def Deterministic_quantization(x, s):
    trans_bit = 0
    # 获取输入向量 x 的维度
    dim = x.shape[0]
    # 计算输入向量 x 的二范数（欧几里得范数）
    xnorm = np.linalg.norm(x)

    # 检查量化级别 s 或范数 xnorm 是否为零
    if s == 0 or xnorm == 0:
        # 如果 s 或 xnorm 有一个为零，则返回相同维度的零填充数组
        return np.zeros(dim, dtype=int)

    # 执行确定量化，使用公式：round(s * |x| / xnorm )

    rounded = np.round(s * np.abs(x) / xnorm).astype(int)

    compressed = (xnorm / s) * np.sign(x) * rounded
    trans_bit = 8
    return compressed, trans_bit


# natural compression (power of 2 for each coordinate)
def natural_compression(x):
    dim = x.shape[0]
    logx = np.ma.log2(np.abs(x)).filled(-15)
    logx_floor = np.floor(logx)
    noise = np.random.uniform(0.0, 1.0, dim)
    leftx = np.exp2(logx_floor)
    rounded = np.floor(np.ma.log2(np.abs(x) + leftx * noise).filled(-15))
    compressed = np.sign(x) * np.exp2(rounded)
    return compressed


def uniform_quantization(v, b0, delta):

    B0 = 2 ** b0  # 量化级别的数量 B_0

    quantized_values = np.zeros(len(v))  # 用于存储每次迭代的量化值


    for i in range(len(v)):
            xnorm = np.linalg.norm(v[i])*10
            if xnorm <= delta / 2:
                quantized_values[i] = 0
            elif xnorm > B0 * delta + delta / 2:
                quantized_values[i] = np.sign(v[i]) * B0 * delta/10
            else:
                for b in range(1, B0 + 1):
                    if (2 * b - 1) * delta / 2 < xnorm <= (2 * b + 1) * delta / 2:
                        quantized_values[i] = np.sign(v[i]) * b * delta/10
                        break
    return quantized_values


# 使用一个数的位数代替传输比特（包括小数部分和整数部分）
def count_digits(number):
    #number是一个dim*1的向量
    total_digits = 0
    for i in range(len(number)):
        num_str = str(abs(number[i]))

        # 如果存在小数点，计算小数点前和小数点后的位数之和
        if '.' in num_str:
            integer_part, decimal_part = num_str.split('.')
            total_digits = total_digits + len(integer_part) + len(decimal_part)
        else:
            # 如果没有小数点，直接统计数字的位数
            total_digits = total_digits + len(num_str)

    return total_digits


def calculate_bits_for_float(x, decimal_precision):
    # 计算整数部分需要的比特数
    integer_part = int(x)
    integer_bits = math.ceil(math.log2(integer_part + 1))

    # 计算小数部分需要的比特数
    # 假设我们使用定点表示，decimal_precision 是小数部分的精度
    fractional_bits = decimal_precision

    # 总比特数
    total_bits = integer_bits + fractional_bits
    return total_bits
