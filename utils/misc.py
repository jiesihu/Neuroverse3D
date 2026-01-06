import torch
import os

def soft_weights_loading(checkpoint_path, args, model):
    tmp = sorted([i for i in os.listdir(checkpoint_path) if '.pth' in i or '.ckpt' in i])[args.checkpoint_index]
    checkpoint_path = os.path.join(checkpoint_path, tmp)
    print('load checkpoints softly from:', checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # 获取 checkpoint 的 state_dict
    checkpoint_state_dict = checkpoint['state_dict']

    # 获取模型的 state_dict
    model_state_dict = model.state_dict()

    # 新建一个匹配 checkpoint 中参数的字典
    filtered_state_dict = {}

    # 遍历 checkpoint 的 state_dict
    for key in checkpoint_state_dict:
        # 去除 "module." 前缀（例如，如果使用了 DataParallel）
        if key in model_state_dict:
            if checkpoint_state_dict[key].shape == model_state_dict[key].shape:
                # 如果名称匹配，则添加到过滤后的字典中
                filtered_state_dict[key] = checkpoint_state_dict[key]
    print('Number of weight loading:',len(filtered_state_dict.keys()))
    model.load_state_dict(filtered_state_dict, strict=False)
    return model
