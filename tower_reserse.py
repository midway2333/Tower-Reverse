import torch
import torch.nn.functional as fc
from torch import nn, Tensor
from typing import Optional, Tuple
from memory import NeuralMemory
from torch.utils.tensorboard import SummaryWriter   # type: ignore
import time

"""
Tower-Reserse, 基于 Tower2 更改
添加长序列记忆支持

原文: https://arxiv.org/abs/2501.00663v1
"""

class FeedForward(nn.Module):
    """全连接层"""
    def __init__(self, d, dff, use_dropout: bool=False):
        """
        参数:
        - d: 输入/输出的维度
        - dff: 前馈网络内部层的维度
        - use_dropout: 是否使用dropout
        """
        super().__init__()
        self.ffn = nn.Sequential(            # 前馈网络
            nn.Linear(d, dff),               # 维度变换
            nn.Dropout(0.05, use_dropout),   # Dropout
            nn.GELU(),                       # 激活函数
            nn.Linear(dff, d),               # 维度变换
            nn.Dropout(0.05, use_dropout)    # Dropout
        )

    def forward(self, inputs: Tensor):
        return self.ffn(inputs)


class RoPE_Emb(nn.Module):
    """RoPE位置编码"""
    def __init__(self, d: int, max_len: int=8192, device: Optional[str]=None):
        """
        RoPE位置编码
        - d: 模型维度
        - max_len: 最大序列长度
        """
        super().__init__()

        self.d = d
        self.max_len = max_len
        self.device = device

        inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2).float().to(device) / d))
        # 计算频率

        self.register_buffer('inv_freq', inv_freq, persistent=False)
        # 注册频率

        self._get_embedding(inv_freq)
        # 预计算

    def _get_embedding(self, inv_freq):
        """预计算位置编码"""
        len_ids = torch.arange(self.max_len, device=self.device)
        # 序列索引

        freqs = torch.outer(len_ids, inv_freq)
        # 计算频率

        emb = torch.cat((freqs, freqs), dim=-1)
        # 复制频率参数, 使复数对共享相同的频率

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        # 频率缓存

    def forward(self) -> Tuple:
        """
        生成RoPE位置编码
        """

        self.cos_cached: Tensor
        self.sin_cached: Tensor

        return (
            self.cos_cached,
            self.sin_cached,
        )

def RoPE_rotate(x: Tensor) -> Tensor:
    """
    RoPE旋转操作
    - x: 输入张量
    """
    x1 = x[..., : x.shape[-1] // 2]   # 取前一半维度
    x2 = x[..., x.shape[-1] // 2 :]   # 取后一半维度
    return torch.cat((-x2, x1), dim=-1)   # 拼接

def RoPE_reshape(x: Tensor) -> Tensor:
    """重塑张量形状"""
    batch, head_num, seq_len, dim = x.shape
    x = x.view(batch, head_num, seq_len, dim//2, 2).transpose(4, 3).reshape(batch, head_num, seq_len, dim)

    return x

def RoPE_apply(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, pos_ids: Tensor):
    """
    应用RoPE编码
    - q: query
    - k: key
    - cos: RoPE cos
    - sin: RoPE sin
    - pos_ids: 位置索引
    """
    cos = cos[pos_ids].unsqueeze(1)   # 按位置索引选择cos值
    sin = sin[pos_ids].unsqueeze(1)   # 按位置索引选择sin值

    q = RoPE_reshape(q)
    # 重塑 Query

    k = RoPE_reshape(k)
    # 重塑 Key

    q_embed = (q * cos) + (RoPE_rotate(q) * sin)
    k_embed = (k * cos) + (RoPE_rotate(k) * sin)
    # 应用旋转位置编码

    return q_embed, k_embed


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


class MLAframe(nn.Module):
    """MLA前置框架"""
    def __init__(self, d, d_pre_head, head_num, max_len: int=8192,
        use_dropout: bool=False, device: Optional[str]=None):
        """
        MLA前置框架
        用于MLA初始化

        参数:
        - d: 输入/输出的维度
        - d_pre_head: 每个头的隐藏层维度 (非RoPE维度)
        - head_num: 头的数量
        - use_dropout: 是否使用dropout
        - device: 计算设备
        """
        super().__init__()

        self.d = d
        self.d_pre_head = d_pre_head
        self.head_num = head_num
        self.use_dropout = use_dropout
        self.device = device
        # 前置传入参数

        self.d_rope = d_pre_head // 2   # 0.5 d_head
        # 计算位置编码维度

        self.dc_v = self.d_pre_head   # 1.0 d_head
        # value维度

        self.dc_kv = self.d_pre_head // 8   # kv Lora维度   0.125 d_head
        self.dc_q = d // 4         # quary Lora维度   0.25 d
        # 低秩压缩的维度
        # DeepSeek 此处 kv 压缩比为 1/14 , q 压缩比为 1/4.7

        self.out_proj = nn.Linear(
            self.head_num * self.dc_v,   # d_head * head_num
            self.d,   # 1.0 d
            bias=False,
        )   # 输出投影

        # ================ quary Lora ================
        self.q_head_dim = self.d_pre_head + self.d_rope   # 1.5 d_head
        # 每个头的quary维度

        self.q_up = nn.Linear(
            self.dc_q,   # 0.25 d
            self.head_num * self.q_head_dim,   # 1.5 d_head * head_num
            bias=False,
        )   # 升维矩阵

        self.q_down = nn.Linear(
            self.d,   # d
            self.dc_q,   # 0.25 d
            bias=False,
        )   # 降维矩阵

        self.q_down_norm = RMS_norm(self.dc_q)

        # =========== key & value Lora ===========
        self.meg_d = self.d_pre_head + self.dc_v   # 1.0 d_head + 0.125 d
        # 合并投影, 便于实现单次完成两者升维

        self.kv_up = nn.Linear(
            self.dc_kv,   # 0.125 d
            self.head_num * self.meg_d,   # 1.0 d_head * head_num + 0.125 d * head_num
            bias=False,
        )   # 升维矩阵

        self.kv_down = nn.Linear(
            self.d,   # 1.0 d
            self.dc_kv + self.d_rope,   # 0.125 d + 0.5 d_head
            bias=False,
        )   # 降维矩阵

        self.kv_norm = RMS_norm(self.dc_kv)

        # ============ RoPE ============
        self.rope = RoPE_Emb(
            self.d_rope,
            max_len=2 * max_len if 2 * max_len > 8192 else 8192,   # 留足余量
            device=device,
        )


class MLA(MLAframe):
    """多头潜在注意力, 通过低秩投影 (LoRA) 压缩 Q/K/V 维度"""
    def __init__(self, d: int, d_pre_head: int, head_num: int, max_len: int,
                 use_dropout: bool=False, device: Optional[str]=None):
        """
        我的想法是尽量减少亢余参数;  
        所以相比于主流实现而言自由度更小, 相应的传参更少

        参数:
        - d: 输入/输出的维度
        - dk_pre_head: 每个头的隐藏层维度 (非RoPE维度)
        - head_num: 头的数量
        - max_len: 最大序列长度
        - use_dropout: 是否使用dropout
        - device: 计算设备
        """
        super().__init__(d, d_pre_head, head_num, max_len, use_dropout, device)

    def without_cache_forward(self, inputs: Tensor, pos_ids: Tensor, mask: Optional[Tensor]=None) -> Tensor:
        """
        不使用缓存的前向传播

        参数:
        - inputs: 输入序列 [batch, seq_len, d]
        - pos_ids: 位置索引
        - mask: 掩码 [seq_len, seq_len]
        - base_len: 基础长度 (通常是全局提示词长度)
        """
        batch_size, seq_len, _ = inputs.size()   # 获得批次与长度

        # ===== quary 计算 =====
        q = self.q_down(inputs)
        q = self.q_down_norm(q)
        q = self.q_up(q)
        # 低秩投影

        q: Tensor = q.view(batch_size, seq_len, self.head_num, self.q_head_dim).transpose(1, 2)
        q_nope, q_rope = torch.split(q, [self.d_pre_head, self.d_rope], dim=-1)
        # 多头拆分 & 维度分割

        # ========= KV 处理 =========
        c_kv = self.kv_down(inputs)
        c_kv, k_rope = torch.split(
            c_kv, [self.dc_kv, self.d_rope], dim=-1
        )   # 分割维度

        c_kv: Tensor = self.kv_norm(c_kv)
        # 归一化 KV

        kv: Tensor = self.kv_up(c_kv)
        # 升维处理

        kv = kv.view(batch_size, c_kv.size(1), self.head_num, self.d_pre_head + self.dc_v).transpose(1, 2)
        k_nope, value = torch.split(kv, [self.d_pre_head, self.dc_v], dim=-1)
        k_rope = k_rope.view(batch_size, c_kv.size(1), 1, self.d_rope).transpose(1, 2)
        # 形状转换 & 矩阵分割

        # ============ RoPE 应用 ============
        cos, sin = self.rope()
        q_rope, k_rope = RoPE_apply(
            q_rope, k_rope, cos, sin, pos_ids,
        )   # 应用 RoPE 编码

        # ============ attention ============
        query = torch.concat(
            [q_nope, q_rope], dim=-1
        )   # 拼接 Query

        key = torch.concat(
            [k_nope, k_rope.expand(-1, self.head_num, -1, -1)], dim=-1
        )   # 拼接 Key

        attn_output = fc.scaled_dot_product_attention(
            query, key, value, attn_mask=mask,
            dropout_p=0.05 if self.use_dropout else 0.0,
        )   # 注意力计算

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.out_proj(attn_output)
        # 变换形状, 输出投影

        return output

    def forward(self, inputs: Tensor, all_pos_ids: Tensor, mask: Optional[Tensor]=None,) -> Tensor:
        """
        - input: 输入序列 [batch, seq_len, d]
        - all_pos_ids: 全序列位置索引
        - inputs_pos_ids: 输入位置索引
        - mask: 掩码 [seq_len, seq_len]
        - base_len: 基础长度 (通常是全局提示词长度)
        """
        return self.without_cache_forward(inputs, all_pos_ids, mask)


class Expert(nn.Module):
    """专家头"""
    def __init__(self, d, dff):
        """标准的 SwiGLU 结构"""
        super().__init__()
        self.Wx = nn.Linear(d, dff, bias=False)
        # 映射线性层

        self.Vx = nn.Linear(d, dff, bias=False)
        # 门控机制

        self.last_linear = nn.Linear(dff, d, bias=False)
        # 输出线性层

    def forward(self, inputs: Tensor):
        """
        - inputs: 输入序列
        """
        Wx = self.Wx(inputs)
        Vx = self.Vx(inputs)
        # 线性映射

        gate = fc.silu(Wx)
        output = gate * Vx
        # 门控机制

        output = self.last_linear(output)
        # 输出线性层

        return output


class MOERouter(nn.Module):
    """路由门逻辑"""
    def __init__(self, d, expert_num, top_k):
        """
        参数:
        - d: 输入维度
        - expert_num: 专家数量
        - top_k: 激活专家数
        """
        super().__init__()
        self.gate = nn.Linear(d, expert_num)   # 路由门
        self.expert_num = expert_num
        self.top_k = top_k

    def forward(self, hidden_states):

        router_logits = self.gate(hidden_states)
        # 计算路由 logits

        routing_probs = fc.softmax(router_logits, dim=-1)
        # softmax
        
        router_weights, selected_experts = torch.topk(
            routing_probs, self.top_k, dim=-1
        )   # 返回 top_k 权重及其专家

        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
        # 权重归一化

        expert_mask = fc.one_hot(selected_experts, num_classes=self.expert_num)
        expert_mask = expert_mask.permute(2, 1, 0)
        # 生成专家掩码, 降低时间复杂度

        return router_logits, router_weights, selected_experts, expert_mask


class SparseMOE(nn.Module):
    def __init__(self, d, dff, expert_num, top_k, init_weights: bool=False):
        """
        稀疏混合专家模型

        参数:
        - d: 输入维度
        - dff: 映射维度
        - expert_num: 专家数量
        - top_k: 激活专家数
        - init_weights: 是否初始化权重
        """
        super().__init__()
        self.d = d
        self.dff = dff
        self.expert_num = expert_num
        self.top_k = top_k
        # 初始化参数

        self.experts = nn.ModuleList([
            Expert(self.d, self.dff) 
            for _ in range(self.expert_num)
        ])  # 添加专家头 

        self.router = MOERouter(self.d, self.expert_num, self.top_k)
        # 路由模块

        if init_weights:   # 初始化权重
            self.apply(generate_init_weights)

    def forward(self, x: Tensor):
        batch_size, seq_len, d = x.shape
        d_states = x.view(-1, d)

        router_logits, router_weights, _, expert_mask = self.router(d_states)
        # 获取路由信息

        router_weights: Tensor   # 方便IDE工作
        final_hidden_states = torch.zeros(
            (batch_size * seq_len, d),
            dtype=d_states.dtype,
            device=d_states.device
        )   # 初始化输出张量

        for expert_idx in range(self.expert_num):   # 遍历每个专家, 检查是否有 token 被分配
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            # 找到需要当前专家处理的 token

            if top_x.shape[0] == 0:
                continue
            # 专家未被 token 选择时, 跳过计算以节省资源

            current_state = d_states[top_x, :]
            # 获取当前专家处理的 token 的输入

            current_hidden_states = expert_layer(current_state) * \
                router_weights[top_x, idx].unsqueeze(-1)
                # 计算加权输出

            final_hidden_states.index_add_(0, top_x, current_hidden_states)
            # 累加最终输出

        final_hidden_states = final_hidden_states.view(batch_size, seq_len, d)
        # 恢复原始形状

        return final_hidden_states, router_logits


class MOE(nn.Module):
    """混合专家模型"""
    def __init__(self, d, dff, share_num, expert_num, top_k, init_weights: bool=False):
        """
        参数:
        - d: 每个专家的输入维度
        - dff: 映射维度
        - share_num: 共享专家数量
        - expert_num: 专家数量
        - top_k: 激活专家数
        - init_weights: 是否初始化权重
        """
        super().__init__()
        self.moe = SparseMOE(d, dff, expert_num, top_k, init_weights)
        # 稀疏混合专家模型

        self.share_experts = nn.ModuleList([
            Expert(d, dff) for _ in range(share_num)
        ])

        if init_weights:   # 初始化权重
            self.apply(generate_init_weights)

    def forward(self, x: Tensor):
        """
        - x: 输入序列 [batch, seq_len, d]
        """

        moe_output, router_logits = self.moe(x)
        # MoE计算

        share_out = [
            expert(x) for expert in self.share_experts
        ]   # 共享专家计算

        share_out = torch.stack(share_out, dim=0).sum(dim=0)
        # 累加共享计算结果
    
        output = share_out + moe_output   # 累加共享与MoE计算结果
        return output                     # 返回输出


class Get_Pos_ids(nn.Module):
    """获得 pos_ids"""
    def __init__(self):
        """创建并获得 pos_ids"""
        super().__init__()

    def forward(self, x: Tensor, _: int=0) -> tuple[Tensor, None]:
        """
        - x: 输入序列 [batch, seq_len, d]
        - _ : 缓存位置 (未使用)
        """
        batch_size, seq_len = x.size(0), x.size(1)
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return pos_ids, None  # [batch_size, seq_len]


class Padding_Mask(nn.Module):
    """填充索引掩码"""
    def __init__(self, padding_idx: int):
        """
        padding_idx: 填充索引
        """
        super().__init__()
        self.padding_idx = padding_idx

    def forward(self, x: Tensor) -> Tensor:
        """
        - x: 输入序列 [batch, seq_len]
        """
        batch_size, seq_len = x.size(0), x.size(1)
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=x.device)
        padding_mask[x == self.padding_idx] = float('-inf')
        padding_mask = padding_mask.reshape(batch_size, 1, 1, seq_len)
        # 创建一个与 x 形状相同的全零矩阵
        # 把 padding 位置设置为-inf
        # 扩展维度

        return padding_mask

class Causal_Mask(nn.Module):
    """因果掩码"""
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        - x: 输入序列 [batch, seq_len]
        """
        batch_size, seq_len = x.size(0), x.size(1)
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
        # 创建上三角掩码, 设置掩码遮掩

        causal_mask = causal_mask.reshape(1, 1, seq_len, seq_len)
        # 扩展维度

        return causal_mask


class Decoder(nn.Module):
    """解码器"""
    def __init__(self, head_num: int, share_num: int, exp_num: int, top_k: int, d: int, dk: int, max_len: int,
        use_dropout: bool=False, init_weights: bool=False, ffn_type: str='ffn'):
        """
        参数:
        - head_num: 注意力头数
        - share_num: 共享专家数
        - exp_num: 专家数量
        - top_k: 激活专家数
        - d: 输入/输出维度
        - dk: 每个头的维度
        - max_len: 最大序列长度
        - use_dropout: 是否使用dropout
        - init_weights: 是否初始化模型
        - ffn_type: 前馈网络类型 (ffn / moe)
        """
        super().__init__()
        self.self_attn = MLA(d, dk, head_num, max_len, use_dropout)
        # 多头自注意力层

        self.get_pos_ids = Get_Pos_ids()
        self.cross_pos_ids = Get_Pos_ids()
        # 获得 pos_ids

        self.self_attn_norm = nn.LayerNorm(d)
        self.cross_attn_norm = nn.LayerNorm(d)
        self.ffn_norm = nn.LayerNorm(d)
        # 层归一化

        assert ffn_type in ['ffn', 'moe'], '请选择正确的前馈网络类型 (ffn / moe)'

        if ffn_type == 'ffn':
            self.ffn = FeedForward(d, 4*d, use_dropout)
            # 前馈网络

        elif ffn_type == 'moe':
            self.ffn = MOE(d, d, share_num, exp_num, top_k, init_weights)
            # 混合专家网络
            # 因为有很多专家, dff的维度较小

        if init_weights:   # 初始化权重
            self.apply(generate_init_weights)

    def forward(
            self, inputs_tuple: Tuple[Tensor, Tensor]
        ) -> Tuple:
        """
        layer_norm 使用 Pre-LN 结构

        参数:
        - inputs_tuple: 输入元组 [input, mask]
        - input: 输入序列 [batch, seq_len, model_dim]
        - mask: 目标序列的掩码 [batch, seq_len]
        """

        inputs, mask = inputs_tuple
        # 解包元组

        # ======= 自注意力阶段 =======
        norm_input = self.self_attn_norm(inputs)   # 归一化输入
        all_pos_ids, inputs_pos_ids = self.get_pos_ids(inputs, 0)
        # 获取位置 ID

        self_attn_output = self.self_attn(
            norm_input,
            all_pos_ids=all_pos_ids,
            mask=mask,
        )   # 自注意力计算

        self_attn_output = inputs + self_attn_output
        attn_output = self_attn_output
        # 残差连接

        # ======= 前馈阶段 =======
        norm_output = self.ffn_norm(attn_output)
        final_output = attn_output + self.ffn(norm_output)
        # 残差连接

        return (final_output, mask)


def generate_init_weights(module: nn.Module):
    """初始化模型权重"""
    if isinstance(module, nn.Linear):   # 线性层
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:   # 线性层偏置
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):   # 嵌入层
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.padding_idx is not None:
            nn.init.zeros_(module.weight[module.padding_idx])

    elif isinstance(module, (nn.LayerNorm, RMS_norm)):   # 层归一化
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)


class Memory_Gate(nn.Module):
    """记忆门逻辑"""
    def __init__(self, dim: int):
        """
        记忆门逻辑
        - dim: 输入维度
        """
        super().__init__()
        self.W_A = nn.Linear(dim, dim)
        self.W_M = nn.Linear(dim, dim)
        # 线性映射

    def forward(self, A: Tensor, M: Tensor) -> Tensor:
        """
        - A: 输入序列 [batch, seq_len, dim]
        - M: 记忆序列 [batch, seq_len, dim]
        """
        attn_seq = self.W_A(A)
        mem_seq = self.W_M(M)
        # 线性映射

        outputs = attn_seq * fc.sigmoid(mem_seq)
        # 计算输出 GLU 门控

        return outputs


class TowerReserse(nn.Module):
    """TowerReserse"""
    def __init__(
        self,
        vocab_size: int,
        dk: int,
        head_num: int,
        share_num: int,
        exp_num: int,
        top_k: int,
        decoder_num: int,
        pad_idx: int,
        max_len: int,
        device: str,
        use_dropout: bool,
        init_weights: bool,
        ffn_type: str,
        train: bool,
        chunk_size: int,
        use_dsconv: bool,
        mem_norm_residual: bool,
        max_batch: int,
    ):
        """
        总模型实现

        参数:
        - vocab_size: 词汇表大小
        - dk: 每个头的维度
        - head_num: 注意力头数
        - share_num: 共享专家数
        - exp_num: 专家数量
        - top_k: 激活专家数
        - decoder_num: 解码器数量
        - pad_idx: 填充索引
        - max_len: 最大序列长度
        - device: 计算设备
        - use_dropout: 是否使用dropout
        - init_weights: 是否初始化模型
        - ffn_type: 前馈网络类型 (ffn / moe)
        - train: 是否处于训练模式 (独立参数, 并非 torch.module.train)
        - use_dsconv: 是否使用深度可分离卷积
        - max_batch: 最大批次大小
        """

        super().__init__()
        self.device = device
        # 初始化设备类型

        self.training = train
        # 是否处于训练模式

        d = dk * head_num
        # 计算输入维度

        self.pad_mask = Padding_Mask(pad_idx).to(device)
        self.casual_mask = Causal_Mask().to(device)
        # 填充索引掩码 / 因果掩码

        self.embed = nn.Embedding(vocab_size, d, padding_idx=pad_idx).to(device)
        # 词表映射

        self.memory_module = NeuralMemory(d, d, chunk_size=chunk_size, use_dsconv=use_dsconv, max_batch=max_batch, 
            mem_norm_residual=mem_norm_residual).to(device)
        # 长序列记忆模块

        self.gate = Memory_Gate(d).to(device)
        # 记忆门逻辑

        self.decoders = nn.ModuleList([
            Decoder(
                head_num=head_num,
                share_num=share_num,
                exp_num=exp_num,
                top_k=top_k,
                d=d,
                dk=dk,
                max_len=max_len,
                use_dropout=use_dropout,
                ffn_type=ffn_type
            ) for _ in range(decoder_num)
        ])   # 解码器

        self.final_norm = RMS_norm(d).to(self.device)
        # 输出归一化, 有利于稳定输出分布

        self.last_linear = nn.Linear(d, vocab_size, bias=False).to(self.device)
        # 输出线性层, 将解码器的输出映射到词汇表的大小

        self.last_linear.weight = self.embed.weight
        # 嵌入层与输出线性层共享权重

        if init_weights:   # 初始化权重
            self.apply(generate_init_weights)

    def forward(self, text_inputs: Tensor, memory_inputs: Tensor | None = None) -> Tensor:
        """
        前向传播
        - text_inputs: text输入序列 [batch, seq_len]
        - memory_inputs: 长序列记忆输入 [batch, seq_len]
        """
        seq_len = text_inputs.size(1)   # 获取序列长度
        embed_output = self.embed(text_inputs)
        # 嵌入层

        if memory_inputs is not None:   # 长序列记忆输入
            memory_inputs = self.embed(memory_inputs)

            if self.training:   # 训练阶段, memory_inputs = history_inputs + text_inputs
                self.memory_module(memory_inputs, store=True)   # 存储记忆
                retrieved = self.memory_module(memory_inputs[:, -seq_len:, :], store=False)   # 检索记忆 # type: ignore
                retrieved = retrieved   # 截取与 text_inputs 相同长度的记忆

            else:   # 推理阶段, memory_inputs = text_inputs
                retrieved = self.memory_module(memory_inputs, store=False)   # 检索记忆
                self.memory_module(memory_inputs, store=True)   # 存储记忆

        else:   # 没有长序列记忆输入
            retrieved = None

        if self.training:   # 训练阶段
            padding_mask = self.pad_mask(text_inputs)
            # 填充掩码

            causal_mask = self.casual_mask(text_inputs)
            # 因果掩码

            text_mask = padding_mask + causal_mask
            # 组合掩码

        else:   # 推理阶段
            text_mask = None

        dec_input = embed_output   # 解码器输入

        for decoder in self.decoders:   # 解码器计算
            dec_input, _ = decoder((dec_input, text_mask))

        decoder_output = dec_input
        # 解码器输出

        if retrieved is not None:
            final_output = self.gate(decoder_output, retrieved)
            # 记忆门逻辑, 将解码器输出与长序列记忆结合

        else:   # 没有记忆输入时
            final_output = decoder_output

        outputs = self.final_norm(final_output)
        outputs = self.last_linear(outputs)
        # 输出线性层

        return outputs


if __name__ == "__main__":
    model = TowerReserse(
        vocab_size=10000,
        dk=64,
        head_num=8,
        share_num=4,
        exp_num=8,
        top_k=2,
        decoder_num=6,
        pad_idx=0,
        max_len=512,
        device='cuda',
        use_dropout=True,
        init_weights=True,
        ffn_type='ffn',
        train=True,
        chunk_size=4,
        use_dsconv=True,
        mem_norm_residual=True,
        max_batch=1,
    ).cuda()

    inputs = torch.randint(
        low=0,
        high=10000,
        size=(1, 64),
        dtype=torch.long
    ).to('cuda')

    mem_inputs = torch.randint(
        low=0,
        high=10000,
        size=(1, 256),
        dtype=torch.long
    ).to('cuda')

    outputs = model(inputs, mem_inputs)
    print(outputs.shape)  # 应该输出 (1, 64, 10000)

