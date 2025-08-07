import torch
from torch import Tensor, nn
import torch.nn.functional as fc
from einops import reduce
from torch.nn.utils import vector_to_parameters
from torch.func import functional_call, vmap, grad   # type: ignore
from typing import Callable


class RMS_norm(nn.Module):
    """均方根层归一化, 相比传统 LayerNorm 有助于梯度稳定性和模型泛化"""
    def __init__(self, hidden_size):
        """
        均方根层归一化 <br>
        hidden_size: 可学习的缩放参数
        """
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        # 定义可学习参数, 初始化

        self.variance_epsilon = 1e-7
        # 防止除零错误

    def forward(self, hidden_states: Tensor):
        input_dtype = hidden_states.dtype                 # 获得原始数据类型
        hidden_states = hidden_states.to(torch.float32)   # 转换成FP32

        variance = hidden_states.pow(2).mean(-1, keepdim=True)   # 沿最后一维计算均方
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # RMS_Norm计算

        return self.weight * hidden_states.to(input_dtype)
        # 还原原始数据类型


class DSConv(nn.Module):
    def __init__(self, dim: int, kernel_size: int=4, stride: int=1, padding: int=1):
        """
        深度可分离卷积层
        - dim: 输入和输出的通道数
        - kernel_size: 卷积核大小
        - stride: 步幅
        - padding: 填充大小
        """
        super().__init__()
        self.depthwise = nn.Conv1d(dim, dim, kernel_size, stride=stride, padding=padding, groups=dim, bias=False)
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.selu = nn.SELU()   # 原论文使用 SELU 激活函数

    def forward(self, x: Tensor) -> Tensor:
        """x: 输入张量 [B, S, D]"""
        x = x.permute(0, 2, 1)   # 转换为 [B, D, S] 以适应 Conv1d
        x = self.pointwise(self.depthwise(x))
        x = self.selu(x)
        return x.permute(0, 2, 1)   # 恢复为 [B, S, D] 形状


class ResidualNorm(nn.Module):
    def __init__(self, dim: int, module: nn.Module):
        """残差连接 + 层归一化模块"""
        super().__init__()
        self.module = module
        self.norm = RMS_norm(dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x + self.module(x))

class Norm(nn.Module):
    def __init__(self, dim: int, module: nn.Module):
        """单层归一化模块"""
        super().__init__()
        self.norm = RMS_norm(dim)
        self.module = module

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.module(x))

def softclamp_grad_norm(tensor: Tensor, max_norm: float) -> Tensor:
    """对梯度执行软截断操作, 避免范数过大"""
    norm = tensor.norm()
    scale = max_norm / (norm + 1e-7)   # 防止除零错误
    scale = torch.clamp(scale, max=1.0)
    return tensor * scale.detach()

class AveragePool(nn.Module):
    def __init__(self, chunk_size):
        """
        平均池化层, 用于将输入张量按指定大小分块并求平均
        - chunk_size: 分块大小
        """
        super().__init__()
        self.chunk_size = chunk_size

    def forward(self, x):
        return reduce(x, 'b (n c) d -> b n d', 'mean', c = self.chunk_size)

class Memory_MLP(nn.Module):
    def __init__(self, dim, depth, rate=4, init_type='zeros'):
        """
        MLP 实现记忆存储

        - dim: 输入维度
        - depth: MLP 中间层数
        - rate: MLP 中间层的扩展率
        - init_type: 初始化方式, 默认为 'zeros'
        - 初始化可选择项: zeros xavier none 
        """
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.rate = rate
        self.init_type = init_type

        assert init_type in ['zeros', 'xavier', 'none'], "init_type must be one of ['zeros', 'xavier', 'none']"
        # 初始化方式检查

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * rate),
            nn.SiLU(),
        )   # 升维投影

        for _ in range(depth):   # 添加 depth 个中间层
            self.mlp.append(nn.Linear(dim * rate, dim * rate))
            self.mlp.append(nn.SiLU())

        self.mlp.append(nn.Linear(dim * rate, dim))
        # 降维投影

        if self.init_type == 'zeros':   # 全零初始化
            for module in self.mlp:
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        elif self.init_type == 'xavier':   # Xavier 初始化
            for module in self.mlp:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        elif self.init_type == 'none':   # torch 默认初始化
            pass
        # 初始化所有 Linear 层

    def _init_memory(self):
        """初始化记忆参数"""
        if self.init_type == 'zeros':   # 全零初始化
            for module in self.mlp:
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        elif self.init_type == 'xavier':   # Xavier 初始化
            for module in self.mlp:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        elif self.init_type == 'none':   # torch 默认初始化
            pass
        # 初始化所有 Linear 层

    def forward(self, input):
        return self.mlp(input)

class NeuralMemory(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dim_head: int,
        depth: int = 2,
        rate: int = 4,
        init_type: str = 'zeros',
        chunk_size: int = 1,
        max_lr = 0.1,
        use_dsconv: bool = False,
        mem_norm_residual: bool = True,
        max_grad_norm: float | None = 1.0,
        max_batch: int = 8,
    ):
        """
        长期神经记忆模块, 支持存储与检索

        参数:
        - dim_model: 输入维度
        - dim_head: 记忆网络输入 / 输出维度
        - depth: MLP 中间层数
        - rate: MLP 中间层的扩展率
        - init_type: MLP 初始化方式, 可选: zeros xavier none
        - chunk_size: 分块大小, 默认为 1
        - max_lr: 自适应学习率的最大值, 默认为 0.1
        - use_dsconv: 是否使用深度可分离卷积, 默认为 False
        - mem_norm_residual: 是否使用残差连接, 默认为 True
        - max_grad_norm: 梯度软限制阈值, 默认为 1.0
        - max_batch: 每次最多维护多少个 batch
        """
        super().__init__()
        self.dim_head = dim_head
        self.chunk_size = chunk_size
        self.max_grad_norm = max_grad_norm
        self.max_lr = max_lr

        self.W_K = nn.Linear(dim_model, dim_head)
        self.W_V = nn.Linear(dim_model, dim_head)
        self.W_Q = nn.Linear(dim_model, dim_head)
        # 注意力投影矩阵

        self.theta_net = nn.Linear(dim_model, 1)   # 学习率
        self.eta_net = nn.Linear(dim_model, 1)     # 动量衰减因子
        self.alpha_net = nn.Linear(dim_model, 1)   # 权重衰减因子
        # 记忆参数网络

        if use_dsconv:
            self.q_conv = DSConv(dim_head, kernel_size=4, stride=1, padding=1)
            self.k_conv = DSConv(dim_head, kernel_size=4, stride=1, padding=1)
            self.v_conv = DSConv(dim_head, kernel_size=4, stride=1, padding=1)
            # 深度可分离卷积

        mem_module = Memory_MLP(dim_head, depth, rate, init_type)

        if mem_norm_residual:   # 残差连接
            self.memory_bank = nn.ModuleList([
            ResidualNorm(dim_head, mem_module) for _ in range(max_batch)
        ])   # 确保样本间记忆独立

        else:   # 无残差
            self.memory_bank = nn.ModuleList([
            Norm(dim_head, mem_module) for _ in range(max_batch)
        ])   # 确保样本间记忆独立

        self.momentum_bank = None  # 为每个 memory 维护动量
            
    def clear_memory(self):
        """清除所有记忆模块的状态"""
        for memory in self.memory_bank.modules():
            if isinstance(memory, Memory_MLP):
                memory._init_memory()
        self.momentum_bank = None

    def loss_fn(self, pred: Tensor, val: Tensor, mask: Tensor) -> Tensor:
        """MSE 损失函数, 输出总 loss"""
        loss = ((pred - val)**2 * mask.unsqueeze(-1)).mean(dim=-1)
        return loss.sum()

    def initialize_momentum_bank(self, batch_size, device):
        """初始化动量缓存"""
        if self.momentum_bank is None or len(self.momentum_bank) != batch_size:
            self.momentum_bank = []
            for i in range(batch_size):   # 添加 batch 个动量缓存
                memory = self.memory_bank[i]
                self.momentum_bank.append(torch.zeros(sum(p.numel() for p in memory.parameters())).to(device))

        else:   # 保持现有的动量状态
            pass

    def reset_momentum_bank(self):
        """重置动量缓存"""
        self.momentum_bank = None

    def store_memories(self, x: Tensor, keys: Tensor, values: Tensor, batch: int):
        """
        记忆模块前向传播
        每个batch的记忆独立存储

        参数:
        - x: 输入序列 [B, S, dim_model]
        - keys: 输入序列的键 [B, S, dim_head]
        - values: 输入序列的值 [B, S, dim_head]
        - batch: 批次大小
        """
        self.momentum: Tensor
        self.memory_param: Tensor
        B, S, D = x.shape
        num_chunks = (S + self.chunk_size - 1) // self.chunk_size   # 计算分块数量
        L = num_chunks * self.chunk_size
        pad_len = L - S   # 计算填充长度
        mask = torch.cat((torch.ones(S), torch.zeros(pad_len)), dim=0).to(x.device)
        mask = mask.reshape(num_chunks, self.chunk_size)
        # 生成 01 掩码 [N, chunk_size]

        x_pad = fc.pad(x, (0, 0, 0, pad_len))
        keys_pad = fc.pad(keys, (0, 0, 0, pad_len))
        values_pad = fc.pad(values, (0, 0, 0, pad_len))
        # 填充处理, 保证最后一个 chunk 大小一致

        self.initialize_momentum_bank(batch, x.device)
        # 动量初始化

        # 1. 分离 batch
        for b in range(batch):   # 分离每个 batch 的记忆
            memory = self.memory_bank[b]
            # 获取当前 batch 的记忆模块

            all_keys = keys_pad[b].view(num_chunks, self.chunk_size, -1)
            all_values = values_pad[b].view(num_chunks, self.chunk_size, -1)
            # 预先分块 KV [N, chunk_size, dim_head]

            momentum = self.momentum_bank[b]   # type: ignore
            # 载入记忆参数 & 初始化动量

            # 2. 分块处理
            for chunk_idx in range(num_chunks):
                key = all_keys[chunk_idx]       # [chunk_size, dim_head]
                value = all_values[chunk_idx]   # [chunk_size, dim_head]

                total_weights, original_params = {}, {}
                for name, param in memory.named_parameters():
                    total_weights[name] = param.clone().detach().requires_grad_(True)
                    original_params[name] = param   # 保存原始参数引用

                # 3. 瞬时惊讶计算
                pred: Tensor = torch.func.functional_call(memory, total_weights, key.unsqueeze(0))   # type: ignore
                # 使用函数式隔离参数 [1, chunk_size, dim_head]

                loss = self.loss_fn(pred, value.unsqueeze(0), mask[chunk_idx])
                # 计算损失 & 应用掩码

                grads = torch.autograd.grad(
                    loss,
                    list(total_weights.values()),
                    retain_graph=False,
                    create_graph=False,
                )   # backward

                surprises = torch.cat([g.reshape(-1) for g in grads])
                # 展平梯度

                if self.max_grad_norm is not None:   # 梯度软限制
                    surprises = softclamp_grad_norm(surprises, self.max_grad_norm)

                theta = torch.sigmoid(self.theta_net(x_pad[b, chunk_idx])).item()   # 学习率
                alpha = torch.sigmoid(self.alpha_net(x_pad[b, chunk_idx])).item()   # 权重衰减因子
                eta = torch.sigmoid(self.eta_net(x_pad[b, chunk_idx])).item()       # 动量衰减因子

                theta = min(theta, self.max_lr)   # 限制学习率

                momentum: Tensor = (eta * momentum - theta * surprises).detach()   # 更新动量
                self.momentum_bank[b] = momentum   # type: ignore

                # 4. 更新记忆
                params = list(memory.parameters())
                flat_params = torch.cat([p.data.view(-1) for p in params])
                new_params = (1 - alpha) * flat_params + momentum
                vector_to_parameters(new_params, memory.parameters())   # 更新参数

    def forward(self, x: Tensor, store: bool=False) -> Tensor | None:
        """
        前向传播:
        - store=True, 执行更新记忆
        - 否则从记忆中获取信息

        参数:
        - x: 输入序列 [B, S, dim_model]
        - store: 是否进行记忆存储

        返回:
        - 记忆输出: 从记忆中检索的值形状为 [B, S, dim_head]
        """
        B, S, _ = x.shape

        # 1. 投影输入序列
        K = self.W_K(x)
        V = self.W_V(x)

        # 2. 记忆处理
        if store:   # 更新记忆
            self.store_memories(x, K, V, B)
            return None

        else:   # 查询记忆
            Q = self.W_Q(x)

            memory_output = torch.zeros(B, S, self.dim_head, device=x.device)
            # 初始化输出张量            

            for batch in range(B):   # 对每个样本使用其对应的记忆模块
                memory = self.memory_bank[batch]
                memory_output[batch] = memory(Q[batch])
                # 查询该样本的记忆

            return memory_output


if __name__ == "__main__":
    # 测试并行化内存模块
    dim_model = 768
    dim_head = 768

    memory_module = NeuralMemory(dim_model, dim_head, max_batch=1, chunk_size=16).to("cuda")

    # 创建一个随机输入序列
    x = torch.randn(1, 2048, dim_model).to("cuda")  # [B, S, dim_model]

    # 存储记忆
    memory_module(x, store=True)

    # 检索记忆
    retrieved: Tensor = memory_module(x, store=False)
    print(retrieved.shape)   # 应该是 [B, S, dim_head]
