import torch

from evoformer import evoformer_base


def inference_evoformer():
    model = evoformer_base().cuda()

    para_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model Size: ", 4 * para_count / 1000 / 1000, " MB")

    model = model.eval()

    node = torch.randn(1, 384, 512, 256).cuda()
    pair = torch.randn(1, 512, 512, 128).cuda()

    print(torch.cuda.memory_summary())

    with torch.no_grad():
        node_o, pair_o = model(node, pair)

    print(torch.cuda.memory_summary())

    print(node_o.shape, pair_o.shape)


if __name__ == '__main__':
    inference_evoformer()
