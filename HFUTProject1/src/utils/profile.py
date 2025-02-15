import torch.autograd.profiler as profiler

def profile_model(model, input_sample):
    """使用PyTorch Profiler分析模型计算开销"""
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA],
        profile_memory=True,
        with_stack=True
    ) as prof:
        model(**input_sample)
    
    print(prof.key_averages().table(
        sort_by="cuda_time_total", 
        row_limit=20
    ))