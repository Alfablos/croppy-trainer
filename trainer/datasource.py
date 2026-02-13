from pathlib import Path
from typing import List

from data import DataSource, DataRow
from common import LabelType


class SmartDocExtendedDataSource(DataSource):
    def __init__(self, name: str, root_path: str, metadata_file: str | None, corner_recess_percentage):
        self.name = name
        self.root_path = Path(root_path)
        self.metadata_file = metadata_file
        self.corner_recess_percentage = corner_recess_percentage
        
        self.images = sorted(list(self.root_path.glob('**/*._in.png')))
        self.labels = sorted(list(self.root_path.glob('**/*._gt.png')))
        
    
    def check(self) -> str | None: # return error as a string
        length_check = len(self.images) == len(self.labels)
        
        if not length_check:
            return f"{self.name} dataset: images and labels should be the same in number!"
    
    def fetch(self, label_type: LabelType) -> List[DataRow]:
        """
            Transforms the dataset into a canonical data source that can be instructed to
            compute masks or coordinates
            Args:
                label_type: coordinates or mask
            Returns:
                List[RowData]: a list of RowData.
        """
        pass



