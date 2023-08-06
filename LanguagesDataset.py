#!/bin/python
from typing import Dict, List, Tuple
from torch.utils.data import Dataset

## Implementation of the Data-Class
class LanguagesDataset(Dataset):
    def __init__(self, rows) -> None:
        self.rows: List[Dict[str, str]] = [row for row in rows]

    def __len__(self) -> int:
        return len(self.rows)
    
    def __getitem__(self, idx) -> Tuple[str, str]:
        row = self.rows[idx]
        return row["labels"], row["text"]
    

# MAIN
def main():
    print("test")

if __name__ == "__main__":
    main()