import json
import os
input_file = "/home/xxx/data/awa2/CUB_200_2011/name_label.txt"
output_file = "/home/xxx/data/awa2/name_str_label.json"
data = {}
with open(input_file, "r") as f:
    for line in f:
        line = line.strip()  # 去除行末的换行符
        name, label = line.split('.')  # 按空格分割行内容为名称和标签
        key = label  # 取名称的第四个字符之后的英文部分作为字典键
        value = name  # 将标签转换为整数作为字典值
        data[key] = value  # 将键值对添加到字典中
with open(output_file, "w") as f:
    json.dump(data, f)
