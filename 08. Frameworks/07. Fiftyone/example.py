import warnings
warnings.filterwarnings("ignore")
import os.path as osp
import argparse
import fiftyone as fo
from fiftyone import zoo as foz
from fiftyone import ViewField as F


class FiftyoneProcessor:
    
    def __init__(
        self, 
        dataset_name, 
        dataset_class, 
        class_map,
        dataset_splits, 
        dataset_type, 
        dataset_download_dir,
        export_dir, 
        max_samples
    ):
        self.dataset_name = dataset_name
        self.dataset_class = dataset_class
        self.class_map = class_map
        self.dataset_splits = dataset_splits
        self.dataset_type = eval("fo.types." + dataset_type)
        self.export_dir = export_dir
        self.max_samples = max_samples
        fo.config.dataset_zoo_dir = dataset_download_dir

    def list_available_datasets(self):
        available_datasets = foz.list_zoo_datasets()
        print("Available datasets:")
        for available_dataset in available_datasets:
            print(available_dataset)
        
    def list_available_datatypes(self):
        available_datatypes = fo.types.__all__
        print("Available datatypes:")
        for datatype in available_datatypes:
            print(datatype)
    
    def download_and_split_dataset(self):
        self.download_dataset()
        self.split_dataset()

    def download_dataset(self):
        self.datasets = {dataset_split: None for dataset_split in self.dataset_splits}
        for dataset_split in self.dataset_splits:
            dataset = fo.zoo.load_zoo_dataset(
                self.dataset_name,
                split=dataset_split,
                classes=self.dataset_class,
                max_samples=None if self.max_samples == -1 else self.max_samples,
                only_matching=False
            )
            self.datasets[dataset_split] = dataset

    def split_dataset(self):
        views = {dataset_split: {} for dataset_split in self.dataset_splits}
        views_s = {dataset_class: views for dataset_class in self.dataset_class}
        for dataset_class in self.dataset_class:
            print("dataset class:", dataset_class)
            for dataset_split in self.dataset_splits:
                print("dataset split:", dataset_split)
                view = self.datasets[dataset_split].filter_labels("ground_truth", F("label") == dataset_class)
                print("data example view:", view.first().ground_truth.detections[0])
                views_s[dataset_class][dataset_split] = view
        for dataset_class in self.dataset_class:
            for dataset_split in self.dataset_splits:
                views_s[dataset_class][dataset_split].export(
                    export_dir=osp.join(self.export_dir, dataset_class),
                    dataset_type=self.dataset_type,
                    classes=self.class_map,  
                    label_field="ground_truth",
                    split=dataset_split
                )

    @classmethod
    def create(cls):
        parser = argparse.ArgumentParser("run 'rm -rf /root/.fiftyone/' if there appears errors of fiftyone's mongodb")
        parser.add_argument("--dataset_name", type=str, default="voc-2012", help="Name of the dataset to download, call .list_available_datasets() to see all available datasets")
        parser.add_argument("--dataset_class", type=list, default=["Person"], help="Class of the dataset to download")
        parser.add_argument("--class_map", type=dict, default={"Person": 0}, help="Map of the dataset class to the label index")
        parser.add_argument("--dataset_splits", type=str, default=["train", "validation"], choices=["train", "validation", "test"], help="Split of the dataset to download")
        parser.add_argument("--dataset_type", type=str, default="YOLOv5Dataset", help="Type of the dataset to download, call .list_available_datatypes() to see all available datatypes")
        parser.add_argument("--dataset_download_dir", type=str, default="/mnt/nfs_docker_volume/training-container-space/mnt/datasets/object-detection-datasets/open-source/fiftyone", help="Directory to download the dataset")
        parser.add_argument("--export_dir", type=str, default="/mnt/nfs_docker_volume/training-container-space/mnt/datasets/object-detection-datasets/open-source/tests", help="Directory to export the dataset")
        parser.add_argument("--max_samples", type=int, default=100, help="Maximum number of samples to download, -1 for all")
        args = parser.parse_args()
        return cls(**vars(args))
    
        
if __name__ == "__main__":
    processor = FiftyoneProcessor.create()
    processor.list_available_datasets()
    processor.list_available_datatypes()
    processor.download_and_split_dataset()




