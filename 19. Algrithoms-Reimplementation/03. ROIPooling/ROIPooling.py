import torch
from torch import nn, Tensor


class ROIPool(nn.Module):
    def __init__(self,
                 feature_map: Tensor,
                 rois: Tensor,
                 output_size
                 ):
        """
        :param feature_map: Tensor. Shape:[N, C, H, W]
        :param rois: Tensor. Shape:[?, 5], 5代表图片index, x1,y1,x2,y2
        :param output_size:
        """
        super(ROIPool, self).__init__()
        self.feature_map = feature_map
        self.rois = rois
        self.output_size = output_size

    def forward(self):
        rois_list = []
        for index in range(self.feature_map.shape[0]):
            rois_list.append([])
        for roi in self.rois:
            rois_list[roi[0]].append(roi[1:])
        output_maps = []
        img_indexes = []
        for img_index, feature_map in enumerate(self.feature_map):
            if not rois_list[img_index]:
                continue
            else:
                for roi in rois_list[img_index]:
                    img_indexes.append(img_index)
                    x1, y1, x2, y2 = roi
                    crop = feature_map[:, y1:y2 + 1, x1:x2 + 1][None]
                    out = nn.AdaptiveAvgPool2d(self.output_size)(crop)
                    output_maps.append(out)
        output = torch.cat(output_maps, 0)
        return img_indexes, output


def generate_fake_rois():
    img_index = torch.randint(0, 15, (5, 1))
    rois_cord_x1y1 = torch.randint(1, 15, (5, 2))
    rois_cord_x2y2 = rois_cord_x1y1 + torch.randint(1, 15, (5, 2))
    rois = torch.cat([img_index, rois_cord_x1y1, rois_cord_x2y2], dim=1)
    return rois


if __name__ == '__main__':
    feature_map = torch.randn(16, 8, 32, 32)
    rois = generate_fake_rois()
    roipool = ROIPool(feature_map, rois, 3)
    img_indexes, output = roipool()
    print(img_indexes)
    print(output.shape)
