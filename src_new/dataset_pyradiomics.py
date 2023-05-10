from pyexpat import features
from tqdm import tqdm
import pickle as pkl
import pandas as pd
import os 
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib 
from torch.utils.data import Dataset,DataLoader
import radiomics
import SimpleITK as itk 
from radiomics import firstorder,shape,glcm,glrlm,glszm

DATASET_PATH = '..'
TRAIN_PATH = 'MICCAI_BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData'
TEST_PATH = 'MICCAI_BraTS2020_ValidationData\\MICCAI_BraTS2020_ValidationData'

class SurTaskRead:
    def __init__(self,root_dir,name='train',ftype=['t1','seg']):
        self.root_dir  = root_dir
        self.ftype     = ftype
        self.name      = name
        self.path = {
            'train':os.path.join(root_dir,DATASET_PATH,TRAIN_PATH),
            'test':os.path.join(root_dir,DATASET_PATH,TEST_PATH)
        }[name]
        
        if name in 'train':
            df = pd.read_csv(os.path.join(self.path,'survival_info.csv'))
            df = df[df['Extent_of_Resection']=='GTR']
            survival_days = df['Survival_days'].tolist()
            selected1     = ~np.array(['ALIVE' in i for i in survival_days])
            ids           = np.array(df['Brats20ID'])[selected1]
            
            self.survival_days = np.array(df['Survival_days'])[selected1]
            self.ages = np.array(df['Age'])[selected1]
            self.filenames = np.array([os.path.join(self.path,filename) for filename in ids])
        else:
            df = pd.read_csv(os.path.join(self.path,'survival_evaluation.csv'))
            ids = df['BraTS20ID'].tolist()
            ages = np.array(df['Age']).astype(np.float32)
            self.ages = ages 
            self.filenames = np.array([os.path.join(self.path,filename) for filename in ids])
        self.ids = ids

        self._init_feat()
        self._init_label()

    def _init_feat(self): 
        '''
            read the number of features and shape
        '''
        index = 0
        filename = [filename for filename in os.listdir(self.filenames[index]) if self.ftype[0] in filename][0]
        image    = nib.load(os.path.join(self.filenames[index],filename)).get_fdata()
        shape    = image.shape

        self.shape = shape
        self.n_feats = len(self.ftype)

    def _init_label(self):
        self.n_classes = None

    def __len__(self):
        return len(self.filenames)

    def _read_feat(self, index):

        path = self.filenames[index]
        filenames = os.listdir(path)
        images = []
        for ftype in self.ftype:
            image_filename = [filename for filename in filenames if ftype in filename][0]
            image          = nib.load(os.path.join(path,image_filename)).get_fdata()
            images.append(image)
        return images,self.ages[index]

    def _read_label(self,index):
        return self.survival_days[index]
    
    def __getitem__(self,index):
        if self.name == 'train':
            feat = self._read_feat(index)
            label = self._read_label(index)
            return feat,label
        elif self.name == 'test':
            feat = self._read_feat(index)
            return (feat,)

class SurCombine:
    def __init__(self,root_dir='.',ftype=['t1','seg']):
        self.root_dir = root_dir

        self.train = SurTaskRead(root_dir,'train',ftype)
        self.test  = SurTaskRead(root_dir,'test',ftype)
        self.shape = self.train.shape 
        self.n_feats   = self.train.n_feats
        self.n_classes = self.train.n_classes
    def __len__(self):
        return len(self.train)

    def collate_fn(self,data):
        feats_mri = []
        feats_age = []
        labels = []
        for d in data:
            feats_mri.append(np.stack(d[0][0],0))
            feats_age.append(d[0][1])
            if len(d) > 1:
                labels.append(int(d[1]))
        feats_mri = torch.tensor(np.stack(feats_mri,0)).float()
        feats_age = torch.tensor(feats_age).float()
        if len(labels) > 0:
            labels = torch.tensor(labels).float()
            return (feats_mri,feats_age),labels
        else:
            return feats_mri,feats_age
    
    def loader(self,name='train',batch_size=1):
        return DataLoader(
            getattr(self,name),
            batch_size=batch_size,
            collate_fn=self.collate_fn
        )


class SurCombineFeaturing(SurCombine):
    def __init__(self,root_dir='.',return_type = 'array'):
        assert return_type in ['tensor','array']
        self.return_type = return_type
        super().__init__(root_dir,['t1','seg'])

    def collate_fn(self, data):

        features       = []
        labels         = []
        for d in data:
            image = itk.GetImageFromArray(d[0][0][0])
            mask  = itk.GetImageFromArray((d[0][0][1]!=0).astype(np.int))
            age   = d[0][1]

            feature = [age]
            for name in ['firstorder','shape','glcm','glrlm','glszm']:
                #'firstorder','shape','glcm','glrlm','glszm'
                obj = globals()[name]
                attr = [i for i in dir(obj) if i.startswith('Radiomics')][0]
                feature.extend(list(getattr(obj,attr)(image,mask).execute().values()))

            feature = np.array(feature)

            if len(d) > 1:
                label = d[1]
            else:
                label = None

            features.append(feature) 
                              
            if label is not None:
                labels.append(label)

        features = np.stack(features,0)
        if len(labels) > 0:
            labels   = np.array(labels)
            if self.return_type == 'tensor':
                features = torch.tensor(features)
                labels   = torch.tensor(labels)
            return features,labels
        else:
            if self.return_type == 'tensor':
                features = torch.tensor(features)
            return features

def give_me_all(loader):
    path = os.path.join(loader.dataset.root_dir,'..','.cache','dataset','all',loader.dataset.name+'.pkl')

    if os.path.exists(path):
        # load
        print('Load Directly')
        data = pkl.load(open(path,'rb'))
    else:
        # process
        if loader.dataset.name=='train' :
            features,labels = [],[]
            for i in tqdm(loader.dataset,desc="Preprocessing"):
                feature,label = loader.collate_fn([i])
                features.append(feature)
                labels.append(label)
            cat_func = {
                np.ndarray:np.concatenate,
                torch.Tensor:torch.cat
            }[type(feature)]
            features = cat_func(features)
            labels   = cat_func(labels)
            data = (features,labels)
        elif loader.dataset.name == 'test':
            features= []
            for i in tqdm(loader.dataset,desc="Preprocessing"):
                feature = loader.collate_fn([i])
                features.append(feature)
            cat_func = {
                np.ndarray:np.concatenate,
                torch.Tensor:torch.cat
            }[type(feature)]
            features = cat_func(features)
            data = features
        else:
            raise Exception(f"Unrecongnized Name {loader.name}")


        path = os.path.join(loader.dataset.root_dir,'..','.cache')
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(path,'dataset')
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(path,'all')
        if not os.path.exists(path):
            os.mkdir(path)
        path =os.path.join(path,loader.dataset.name+'.pkl')
        # save
        pkl.dump(data,open(path,'wb'))
        print('Dump Finish')
    return data

if __name__ == '__main__':
    dataset = SurCombine(ftype=['t1'])
    print(dataset.__class__.__name__)
    print(dataset.shape)
    print(len(dataset.train))
    print(len(dataset.test))
    # print(next(iter(dataset.loader('train')))[0].shape)
    # print(next(iter(dataset.loader('train')))[1].shape)
    loader = dataset.loader('train',batch_size=2)
    (x1,x2),y = next(iter(loader))
    print('x1',x1.shape,x1.dtype)
    print('x2',x2.shape,x2.dtype)
    print('y',y.shape,y.dtype)