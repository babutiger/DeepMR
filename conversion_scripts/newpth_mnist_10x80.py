# 把普通训练好的 pth 文件 转换成 beta-crown 需要的 pth 文件 2025年01月14日21:42:54

# 加载预训练模型的权重：在实例化 OriginalNeuralNetwork 类之后，确保你加载了相应的权重。
# 避免在每次运行代码时重新初始化权重：通过直接从文件中加载已保存的模型权重。
# 自动生成 mapping 字典

import torch
from torch import nn

# 原始模型的定义
class OriginalNeuralNetwork(nn.Module):
    def __init__(self):
        super(OriginalNeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 10)
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x

# 修改后的模型，定义为 nn.Sequential，添加 nn.Flatten()
def create_modified_model():
    model = nn.Sequential(
        nn.Flatten(),  # 添加 Flatten 层
        nn.Linear(784, 80),
        nn.ReLU(),
        nn.Linear(80, 80),
        nn.ReLU(),
        nn.Linear(80, 80),
        nn.ReLU(),
        nn.Linear(80, 80),
        nn.ReLU(),
        nn.Linear(80, 80),
        nn.ReLU(),
        nn.Linear(80, 80),
        nn.ReLU(),
        nn.Linear(80, 80),
        nn.ReLU(),
        nn.Linear(80, 80),
        nn.ReLU(),
        nn.Linear(80, 80),
        nn.ReLU(),
        nn.Linear(80, 80),
        nn.ReLU(),
        nn.Linear(80, 10)
    )
    return model


# 加载原来的模型参数
original_model_path = '../mnist_net_new_10x80.pth'
original_state_dict = torch.load(original_model_path)

# 创建原始模型实例并加载权重
old_model = OriginalNeuralNetwork()
old_model.load_state_dict(original_state_dict)

# 创建新的模型实例
new_model = create_modified_model()

# 自动生成 mapping 字典
mapping = {}
for i, layer in enumerate(old_model.linear_relu_stack):
    if isinstance(layer, nn.Linear):
        print(f"i:{i}")
        # 这里原先搞错了，导致有些层没有处理，只处理了0 4 8 这种，因为我乘以2了，而i本身就是0 2 4 这种 2025年01月14日21:14:23
        mapping[f'linear_relu_stack.{i}.weight'] = f'{i + 1}.weight'
        mapping[f'linear_relu_stack.{i}.bias'] = f'{i + 1}.bias'



# 将原模型的参数转换为新模型的格式
new_state_dict = {}
for old_key in mapping.keys():
    new_key = mapping[old_key]
    if old_key in original_state_dict:
        new_state_dict[new_key] = original_state_dict[old_key]



# new_state_dict = {}
# # 根据映射关系重新命名参数
# for old_key, new_key in mapping.items():
#     if old_key in original_state_dict:
#         new_state_dict[new_key] = original_state_dict[old_key]


# 将修正后的参数加载到新模型
new_model.load_state_dict(new_state_dict, strict=False)

# 保存新的模型为 .pth 文件
new_model_path = '../mnist_net_new_10x80_check_for_bcrown.pth'
torch.save(new_model.state_dict(), new_model_path)

print(f"New model saved to {new_model_path} with nn.Flatten() added.")

# 打印原始模型的结构和参数信息
print("\nOriginal Model Structure:")
print(old_model)

print("\nOriginal Model Layer-wise Parameters:")
for name, param in old_model.named_parameters():
    print(f"{name}: {param.size()}")

# 打印修改后的模型结构
print("\nModified Model Structure:")
print(new_model)

# 打印修改后的模型的每一层参数的维度
print("\nModified Model Layer-wise Parameters:")
for name, param in new_model.named_parameters():
    print(f"{name}: {param.size()}")

# 打印新旧模型的所有权重参数和偏置参数
print("\nOriginal Model Weights and Biases:")
for name, param in old_model.named_parameters():
    if 'weight' in name or 'bias' in name:
        print(f"{name}: {param.data}")

print("\nModified Model Weights and Biases:")
for name, param in new_model.named_parameters():
    if 'weight' in name or 'bias' in name:
        print(f"{name}: {param.data}")
