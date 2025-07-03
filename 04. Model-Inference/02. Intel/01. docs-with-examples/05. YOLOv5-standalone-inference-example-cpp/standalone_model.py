from pathlib import Path
import yaml
import torch
import torch.nn as nn
import numpy as np
from openvino.runtime import Core, Layout, get_batch
from standalone_utils import LOGGER


class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False, fuse=True):
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        LOGGER.info(f'Loading {w} for OpenVINO inference...')
        ie = Core()
        w = next(Path(w).glob('*.xml'))  # get *.xml file from *_openvino_model dir
        network = ie.read_model(model=w, weights=Path(w).with_suffix('.bin'))
        network.get_parameters()[0].set_layout(Layout("NCHW"))
        batch_dim = get_batch(network)
        batch_size = batch_dim.get_length()
        executable_network = ie.compile_model(network, device_name="CPU")  # device_name="MYRIAD" for Intel NCS2
        stride, names = self._load_metadata(Path(w).with_suffix('.yaml'))  # load metadata
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        im = im.cpu().numpy()  # FP32
        y = list(self.executable_network([im]).values())
        return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
    
    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x
        
    def _load_metadata(self, file=Path('path/to/meta.yaml')):
        with open(file, errors='ignore') as f:
            d = yaml.safe_load(f)
        return d['stride'], d['names']  # assign stride, names