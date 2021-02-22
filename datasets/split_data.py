import os
import shutil
import numpy as np
class Dataset:
    def __init__(self,path,split_ratio = 0.9):
        self.path = path
        self.code= os.listdir(path)
        self.split_ratio = split_ratio
        
    def get_code_num(self):
        return {c:len(os.listdir(os.path.join(self.path,c))) for c in self.code}
    def get_img_list(self,folder):
        res=[]
        for f in os.listdir(folder):
            if(f.endswith(('.bmp','.jpg','.png','.jpeg'))):
                res.append(os.path.join(folder, f))
        return res
    
    def train_dev_split(self, X):
        train_len = int(round(len(X)*self.split_ratio))
        return X[0:train_len], X[train_len:None]

    def shuffle(self,X):
        X = np.array(X)
        randomize = np.arange(len(X))
        np.random.shuffle(randomize)
        return X[randomize]
               
    
    def save_img_list(self,img_list:list , des:str):
        for f in img_list:
            des_path = os.path.join(des ,os.path.split(f)[1] )
            shutil.copy(f, des_path)
            
            
            
def main():
    src_path = 'datasets/raw_data'
    des_train_path = 'datasets/train-0.8'
    des_val_path = 'datasets/val-0.2'
    os.makedirs(des_train_path ,exist_ok=True)
    os.makedirs(des_val_path ,exist_ok=True)
    dataset = Dataset(src_path,split_ratio=0.8)
    print(dataset.get_code_num())

    for c in dataset.code:
        img_list = dataset.get_img_list(os.path.join(src_path,c))
        img_list = dataset.shuffle(img_list)
        train_x , val_x = dataset.train_dev_split(img_list)
        os.makedirs(os.path.join(des_train_path , c),exist_ok=True)
        os.makedirs(os.path.join(des_val_path , c),exist_ok=True)
        dataset.save_img_list(train_x, os.path.join(des_train_path , c))
        dataset.save_img_list(val_x, os.path.join(des_val_path,c))

        
if __name__ == "__main__":
    main()
    