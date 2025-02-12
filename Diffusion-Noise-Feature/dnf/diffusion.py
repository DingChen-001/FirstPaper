import math
import torch
import torch.nn as nn

# 定义了一个函数get_timestep_embedding，接受timesteps和embedding_dim作为参数
# 该函数的作用是将时间步timesteps编码为高维向量，常用于扩散模型中。
# 通过将每个时间步映射到sin和cos的组合，并利用不同的频率基底，生成的嵌入可以捕捉到不同时间步的特征，便于模型学习噪声扩散过程中的信息。
# 这种编码方式类似于Transformer的位置编码，但应用于时间步而非位置。
def get_timestep_embedding(timesteps, embedding_dim):

    assert len(timesteps.shape) == 1 # 确保输入的timesteps是一个一维张量，即形状为 (batch_size,) 或类似
    
    half_dim = embedding_dim // 2 # 将嵌入维度除以2，用于生成频率序列。后续会将sin和cos结果拼接，总维度恢复为embedding_dim
    emb = math.log(10000) / (half_dim - 1) # 计算一个对数值作为频率基底，用于生成变化的频率
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb) # 生成一个递减指数序列，每个元素对应不同的频率。arange生成0到half_dim-1的整数
    emb = emb.to(timesteps.device) # 将计算得到的频率向量移动到与timesteps相同的设备（如GPU或CPU）
    emb = timesteps.float()[:, None] * emb[None, :] # 扩展timesteps的维度并在第二个维度上与频率向量相乘，形状为(batch_size, half_dim)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1) # 分别计算sin和cos，并在第一个维度拼接，得到形状为(batch_size, embedding_dim)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0)) # 如果embedding_dim是奇数，则进行零填充以匹配维度
    return emb # 返回最终的时间步嵌入向量，形状为(batch_size, embedding_dim)



# 定义一个非线性激活函数，使用Swish激活函数
def nonlinearity(x):
    # Swish激活函数的实现：x乘以sigmoid(x)
    return x * torch.sigmoid(x)


# 定义一个归一化层，使用GroupNorm进行归一化操作
def Normalize(in_chnnels):
    # 使用torch.nn.GroupNorm实现归一化
    # num_groups=32：将输入通道分为32个组
    # num_channels=in_channels：输入的总通道数
    # eps=1e-6：防止除零错误的极小值
    # affine=True：允许仿射变换（即带有可学习的缩放和偏置参数）
    return torch.nn.GroupNorm(num_roups=32, num_chnnels=in_chnnels, eps=1e-6, affine=True)


# 定义一个上采样模块，用于将输入特征图的分辨率提高一倍
class Upsample(nn.Module):
    def __init__(self, in_chnnels, with_conv):
        super().__init__()
        self.with_conv = with_conv  # 是否使用卷积操作

        # 如果with_conv为True，则定义一个卷积层
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_chnnels,  # 输入通道数
                in_chnnels,  # 输出通道数（保持不变）
                kernel_size=3,  # 卷积核大小为3x3
                stride=1,       # 步长为1
                padding=1        # 填充为1，保证输出特征图的尺寸正确
            )

    def forward(self, x):
        # 使用最近邻插值将输入特征图的分辨率提高一倍
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0,  # 缩放因子为2.0，即将高度和宽度各放大2倍
            mode="nearest"     # 使用最近邻插值方式
        )

        # 如果with_conv为True，则对上采样的结果进行卷积操作
        if self.with_conv:
            x = self.conv(x)

        return x


# 定义一个下采样模块，用于将输入特征图的分辨率降低一半
class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):  # 初始化方法，接受输入通道数和是否使用卷积操作
        super().__init__()                      # 调用父类Module的初始化方法
        self.with_conv = with_conv              # 是否在下采样后应用卷积层

        if self.with_conv:                       # 如果需要进行卷积操作，则定义一个3x3卷积层
            self.conv = torch.nn.Conv2d(
                in_channels,    # 输入通道数
                in_channels,    # 输出通道数（保持不变）
                kernel_size=3, # 卷积核大小为3x3
                stride=2,      # 步长为2，下采样因子
                padding=0      # 填充为0，默认不填充
            )

    def forward(self, x):                     # 前向传播方法，处理输入张量x
        if self.with_conv:                    # 如果需要进行卷积操作
            pad = (0, 1, 0, 1)                # 定义手动填充的尺寸，(left, right, top, bottom)，这里左右各不填充和填充1个像素？ 等待，这可能需要重新审视。 
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)  # 对输入张量x进行零填充
            x = self.conv(x)                  # 应用卷积层进行下采样
        else:                                 # 如果不需要卷积操作，则使用平均池化进行下采样
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)  # 使用2x2的平均池化，步长为2

        return x                              # 返回下采样后的张量


# 定义一个残差块类，继承自nn.Module
class ResnetBlock(nn.Module):
    def __init__(self, *, in_chnnels, out_chnnels=None, conv_shortcut=False,
                 dropout, temb_chnnels=512):
        super().__init__()  # 调用父类的初始化方法
        
        self.in_chnnels = in_chnnels    # 输入通道数
        self.out_chnnels = out_chnnels if out_chnnels is not None else in_chnnels  # 输出通道数，默认与输入相同
        
        self.use_conv_shortcut = conv_shortcut  # 是否使用卷积捷径连接
        
        # 第一层归一化和卷积
        self.norm1 = Normalize(in_chnnels)    # 归一化层，作用于输入特征图
        self.conv1 = torch.nn.Conv2d(
            in_chnnels,     # 输入通道数
            out_chnnels,    # 输出通道数
            kernel_size=3,  # 卷积核大小为3x3
            stride=1,       # 步长为1，保持特征图尺寸不变
            padding=1       # 填充为1，确保输出尺寸与输入一致
        )
        
        # 时间嵌入投影层
        self.temb_proj = torch.nn.Linear(
            temb_chnnels,  # 时间嵌入的维度
            out_chnnels    # 投影到与特征图相同通道数的空间
        )
        
        # 第二层归一化、激活函数和卷积
        self.norm2 = Normalize(out_chnnels)   # 归一化层，作用于第一次卷积后的特征图
        self.dropout = torch.nn.Dropout(dropout)  # Dropout层，用于防止过拟合
        
        self.conv2 = torch.nn.Conv2d(
            out_chnnels,    # 输入通道数
            out_chnnels,    # 输出通道数（保持不变）
            kernel_size=3,  # 卷积核大小为3x3
            stride=1,       # 步长为1，保持特征图尺寸不变
            padding=1       # 填充为1，确保输出尺寸与输入一致
        )
        
        # 根据输入和输出通道数是否不同来决定是否添加捷径连接
        if self.in_chnnels != self.out_chnnels:
            if self.use_conv_shortcut:  # 如果使用卷积捷径
                self.conv_shortcut = torch.nn.Conv2d(
                    in_chnnels,    # 输入通道数
                    out_chnnels,   # 输出通道数
                    kernel_size=3,  # 卷积核大小为3x3
                    stride=1,       # 步长为1，保持特征图尺寸不变
                    padding=1       # 填充为1，确保输出尺寸与输入一致
                )
            else:   # 否则使用1x1的卷积捷径（线性变换）
                self.nin_shortcut = torch.nn.Conv2d(
                    in_chnnels,    # 输入通道数
                    out_chnnels,   # 输出通道数
                    kernel_size=1,  # 卷积核大小为1x1，仅调整通道数
                    stride=1,       # 步长为1
                    padding=0       # 不填充
                )

    def forward(self, x, temb):
        h = x  # 将输入特征图赋值给h
        
        h = self.norm1(h)            # 第一次归一化操作
        h = nonlinearity(h)          # 应用非线性激活函数
        h = self.conv1(h)            # 第一次卷积操作
        
        # 投影时间嵌入并进行加法操作
        temb_proj = self.temb_proj(nonlinearity(temb))  # 将时间嵌入通过非线性和投影层变换
        # 扩展维度以便与特征图相加
        h += temb_proj[:, :, None, None]  # 在空间维度上扩展，使形状匹配
        
        h = self.norm2(h)            # 第二次归一化操作
        h = nonlinearity(h)          # 应用非线性激活函数
        h = self.dropout(h)          # Dropout层，防止过拟合
        h = self.conv2(h)            # 第二次卷积操作
        
        # 根据输入和输出通道数是否不同来调整捷径连接
        if self.in_chnnels != self.out_chnnels:
            if self.use_conv_shortcut:  # 使用3x3的卷积捷径
                x = self.conv_shortcut(x)
            else:                       # 使用1x1的卷积捷径
                x = self.nin_shortcut(x)
        
        return x + h  # 将调整后的输入与经过两次卷积的特征图相加，得到最终输出


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        # 定义归一化层
        self.norm = Normalize(in_channels)
        
        # 定义查询、键和值的卷积层，输出通道数与输入相同，使用1x1卷积不改变空间尺寸
        self.q = torch.nn.Conv2d(
            in_channels,    # 输入通道数
            in_channels,    # 输出通道数（保持不变）
            kernel_size=1,  # 卷积核大小为1x1
            stride=1,       # 步长为1，保持尺寸不变
            padding=0      # 不填充
        )
        
        self.k = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        self.v = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # 定义输出投影层，用于调整特征图的通道数
        self.proj_out = torch.nn.Conv2d(
            in_channels,    # 输入通道数
            in_channels,    # 输出通道数（保持不变）
            kernel_size=1,
            stride=1,
            padding=0
        )
        
    def forward(self, x):
        h_ = x  # 将输入特征图赋值给h_
        h_ = self.norm(h_)  # 应用归一化层
        
        # 生成查询、键和值向量
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        
        # 计算注意力权重
        b, c, h, w = q.shape  # 获取批大小、通道数、高度和宽度
        
        # 将查询、键、值重塑为二维形状，便于计算注意力
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # 转置，得到形状 [b, hw, c]
        
        k = k.reshape(b, c, h * w)  # 形状为 [b, c, hw]
        v = v.reshape(b, c, h * w)  # 形状为 [b, c, hw]
        
        # 计算注意力权重矩阵
        w_ = torch.bmm(q, k)  # [b, hw, hw]，计算点积相似度
        
        # 缩放处理，防止数值过大或过小
        scale_factor = int(c) ** (-0.5)
        w_ *= scale_factor
        
        # 应用Softmax函数，对行进行归一化
        w_ = torch.nn.functional.softmax(w_, dim=2)
        
        # 计算加权值向量
        h_ = torch.bmm(v, w_)  # [b, c, hw] 与 [b, hw, hw] 相乘，得到新的特征表示
        
        # 将结果重塑回原来的形状，并通过输出投影层进行线性变换
        h_ = h_.reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        
        # 残差连接，将原始输入与注意力输出相加
        return x + h_


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()  # 调用父类的初始化方法
        self.config = config  # 存储配置对象
        
        # 从配置中提取模型参数
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult) # 输入通道数，输出通道数, 通道数乘法器元组，用于确定不同分辨率下的通道数量
        num_res_blocks = config.model. num_res_blocks  # 每个分辨率下残差块的数量
        attn_resolutions = config.model.attn_resolutions  # 使用注意力机制的分辨率列表
        dropout = config.model.dropout    # dropout 概率
        in_channels = config.model.in_channels   # 输入通道数
        resolution = config.data.image_size      # 输入图像的大小（高和宽）
        resamp_with_conv = config. model.resamp_with_conv  # 是否使用卷积进行上采样和下采样
        num_timesteps = config.diffusion.num_diffusion_timesteps
        
        if config.model.type == 'bayesian':  # 如果模型类型是贝叶斯，则添加 logvar 参数
            self.logvar = nn.Parameter(torch.zeros(num_timesteps))  # 初始化为全零
            
        self.ch = ch                            # 设置基础通道数
        self.temb_ch = self.ch *4             # 时间嵌入层的通道数，通常是基础通道数的四倍
        self.num_resolutions = len(ch_mult)    # 计算分辨率的数量
        self. num_res_blocks = num_res_blocks  # 存储残差块数量
        self.resolution = resolution           # 设置输入图像的大小
        self.in_channels = in_channels        # 设置输入通道数
        
        # 时间嵌入层，将时间步编码为高维向量
        self. temb = nn.Module()
        self. temb.dense = nn.ModuleList([
            torch. nn.Linear(self.ch,  # 第一层：从基础通道数映射到四倍的基础通道数
                            self.temb_ ch),
            torch. nn.Linear(self.temb_ch,  # 第二层：保持四倍的基础通道数
                            self.temb_ ch)
        ])
        
        # 下采样部分，将输入图像逐步压缩特征图的尺寸和增加通道数
        self.conv_in = torch.nn.Conv2d(  # 输入卷积层，从输入通道数转换到基础通道数
            in_channels,
            self.ch,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        curr_res = resolution   # 当前分辨率初始化为配置中的图像大小
        in_ch_mult = (1,) + ch_mult  # 用于确定每个级别输入的通道数，初始为1倍基础通道
        
        self. down = nn.ModuleList()  # 下采样模块列表
        block_in = None             # 当前块的输入通道数
        
        for i_level in range(self.num_resolutions):  # 遍历每个分辨率级别
            block = nn.ModuleList()  # 残差块列表
            attn = nn.ModuleList()   # 注意力模块列表
            
            block_in = ch * in_ch_mult[i_level]  # 计算当前级别的输入通道数
            block_out = ch * ch_mult[i_level]     # 计算当前级别的输出通道数
            
            for i_block in range(self.num_res_blocks):  # 遍历每个残差块
                # 添加一个残差块，传入当前的输入通道数、输出通道数等参数
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,  # 时间嵌入层的通道数
                        dropout=dropout
                    )
                )
                block_in = block_out  # 更新输入通道数为当前块的输出
                
            if curr_res in attn_resolutions:  # 如果当前分辨率在注意力分辨率列表中，则添加注意力模块
                attn.append(AttnBlock(block_in))
                
            down_module = nn.Module()  # 创建下采样模块
            down_module.block = block    # 设置残差块列表
            down_module.attn = attn      # 设置注意力模块列表
            
            if i_level != self.num_resolutions -1:  # 如果不是最后一个级别，则添加下采样层
                down_module.downsample = Downsample(block_in, resamp_with_conv)
                
            curr_res = curr_res //2    # 将当前分辨率减半
            
            self. down.append(down_module)  # 将当前级别的模块添加到下采样列表中
            
        # 中间层，通常包含一些残差块和注意力机制
        self. mid = nn.Module()
        self. mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,  # 输出通道数与输入相同
            temb_channels=self.temb_ch,
            dropout=dropout
        )
        self. mid.attn_1 = AttnBlock(block_in)  # 添加注意力模块
        self. mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout
        )
        
        # 上采样部分，逐步恢复特征图的尺寸和减少通道数
        self. up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):  # 倒序遍历分辨率级别
            block = nn.ModuleList()  # 残差块列表
            attn = nn.ModuleList()   # 注意力模块列表
            
            block_out = ch * ch_mult[i_level]  # 当前级别的输出通道数
            skip_in = ch * ch_mult[i_level]  # 跳跃连接的输入通道数
            
            for i_block in range(self.num_res_blocks +1):  # 比下采样多一个残差块
                if i_block == self. num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]  # 最后一个块的跳跃连接输入通道数不同
                
                # 添加残差块，传入当前的输入通道数、输出通道数等参数
                block.append(
                    ResnetBlock(
                        in_channels=block_in + skip_in,  # 输入是上一层和跳跃连接的特征图
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout
                    )
                )
                
                block_in = block_out  # 更新输入通道数
                
            if curr_res in attn_resolutions:  # 如果当前分辨率在注意力列表中，则添加注意力模块
                attn.append(AttnBlock(block_in))
                
            up_module = nn.Module()  # 创建上采样模块
            up_module.block = block    # 设置残差块列表
            up_module.attn = attn      # 设置注意力模块列表
            
            if i_level !=0:           # 如果不是第一个级别，则添加上采样层
                up_module.upsample = Upsample(block_in, resamp_with_conv)
                
            curr_res = curr_res *2   # 将当前分辨率加倍
            
            self.up.insert(0, up_module)  # 将当前级别的模块插入到上采样列表的最前面，以保持顺序
        
        # 最终的归一化和输出卷积层
        self.norm_out = Normalize(block_in)  # 归一化层
        self. conv_out = torch.nn.Conv2d(  # 输出卷积层，将通道数从 block_in 转换到 out_ch
            block_in,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
    def forward(self, x, t):
        assert x. shape[2] == x.shape[3] == self.resolution  # 验证输入图像的尺寸是否正确
        
        # 时间嵌入层，处理时间步 t
        temb = get_timestep_embedding(t, self.ch)  # 获取时间嵌入向量
        temb = self. temb.dense[0](temb)             # 第一层全连接
        temb = nonlinearity(temb)                    # 非线性激活函数
        temb = self. temb.dense[1](temb)             # 第二层全连接
        
        # 下采样部分，逐层处理特征图
        hs = [self.conv_in(x)]  # 初始下采样后的特征图列表
        
        for i_level in range(self.num_resolutions):  # 遍历每个分辨率级别
            for i_block in range(self. num_res_blocks):  # 遍历每个残差块
                h = self. down[i_level]. block[i_block](hs[-1], temb)  # 应用当前残差块
                if len(self.down[i_level]. attn) >0:  # 如果有注意力模块，则应用
                    h = self. down[i_level]. attn[i_block](h)
                hs.append(h)  # 将特征图添加到列表中
            
            if i_level != self.num_resolutions -1:  # 如果不是最后一个级别，进行下采样
                hs.append(self. down[i_level].downsample(hs[-1]))
        
        # 中间层处理
        h = hs[-1]  # 取最新的特征图
        h = self. mid.block_1(h, temb)  # 第一个残差块
        h = self. mid.attn_1(h)          # 注意力模块
        h = self. mid.block_2(h, temb)   # 第二个残差块
        
        # 上采样部分，逐步恢复特征图的尺寸和通道数
        for i_level in reversed(range(self.num_resolutions)):  # 倒序遍历分辨率级别
            for i_block in range(self. num_res_blocks +1):     # 遍历每个残差块
                if i_block ==0:   # 如果是第一个块，不需要跳跃连接
                    h = self.up[i_level].block[i_block](h, temb)
                else:
                    # 跳跃连接：将当前特征图与下采样时的对应特征图拼接
                    h = self. up[i_level]. block[i_block](
                        torch.cat([h, hs.pop()], dim=1),  # 沿通道维度拼接
                        temb
                    )
                if len(self.up[i_level]. attn) >0:   # 如果有注意力模块，则应用
                    h = self.up[i_level].attn[i_block](h)
                
            if i_level !=0:       # 如果不是第一个级别，进行上采样
                h = self. up[i_level].upsample(h)
        
        # 最终的归一化和输出
        h = self. norm_out(h)     # 应用归一化层
        h = nonlinearity(h)      # 非线性激活函数
        h = self. conv_out(h)    # 输出卷积层，生成最终图像
        
        return h  # 返回模型输出
