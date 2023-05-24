import torch
import PIL.Image as Image



'''Load Images from text file

The class inherits from torch.utils.data.Dataset
and is used to load images from a text file.
The text file contains two columns, the first column
is recorded the path to the image, and the second column
is recorded the label of the image.
'''

class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, train_fpath: str, class_fpath: str, transform: str=None) -> None:
        # load training image data
        with open(train_fpath, 'r') as trainfh:
            x = []
            y = []
            for line in trainfh:
                words = line.rstrip().split('\t')
                x.append(words[0])
                y.append(words[1])
        # load class labels
        with open(class_fpath, 'r') as classfh:
            classes = {}
            for line in classfh:
                words = line.rstrip().split('\t')
                classes[words[0]] = words[1]
        # save data to the private variables
        self.x = x
        self.y = y
        self.classes = classes
        self.transform = transform


    def __getitem__(self, i):
        img = PIL.Image.open(self.x[i]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)    
        label = self.classes[self.y[i]]
        return img, label


    def __len__(self):
        return len(self.x)
    
    




if __name__ == '__main__':
    pass
