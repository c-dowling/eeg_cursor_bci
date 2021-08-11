from utils import concat_datasets
from torch.utils.data import DataLoader
import json


def main():
    with open("params.json") as fp:
        params = json.load(fp)

    dataset = concat_datasets(params["in_dir"])

    dataLoader = DataLoader(dataset,
                            batch_size=params['batch_size'],
                            num_workers=2,
                            shuffle=True)

    for idx, (data,label) in enumerate(dataLoader):
        print(data.shape,label.shape)
        print("-------------------------")



if __name__ == "__main__":
    main()
    