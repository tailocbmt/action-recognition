import torch

class TemporalSequenceCrop(object):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.short_side_scale``. The size
    parameter is chosen randomly in [min_size, max_size].
    """

    def __init__(self, size):
        super().__init__()
        self.size = size

    def __call__(self, frames, index: int=0) -> torch.Tensor:
        time_index = index * self.size

        buffer = frames[time_index:time_index + self.size]
        
        for index in buffer:
            if len(buffer) >= self.size:
                break
            buffer.append(index)

        out = torch.stack(buffer, 0)
        out = out.permute((3, 0, 1, 2))
        return out