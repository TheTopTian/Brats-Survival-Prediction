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
from radiomics import firstorder,shape,glcm,glrlm,glszm,ngtdm,gldm
import csv

DATASET_PATH = '..'
TRAIN_PATH = 'MICCAI_BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData'
TEST_PATH = 'MICCAI_BraTS2020_ValidationData\\MICCAI_BraTS2020_ValidationData'

class SurTaskRead:
    def __init__(self,root_dir,name='train',ftype=['t1','t1ce','t2','flair','seg']):
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
            filename    = [filename for filename in filenames if ftype in filename][0]
            print(filename)
            image       = nib.load(os.path.join(path,filename)).get_fdata()
            images.append(image)
        print(np.shape(images))
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
    def __init__(self,root_dir='.',ftype=['t1','t1ce','t2','flair','seg']):
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
        super().__init__(root_dir,['t1','t1ce','t2','flair','seg'])

    def collate_fn(self, data):
        def center(x):
            _x,_y,_z = np.meshgrid(*[np.arange(i) for i in x.shape])
            _x = _x[x].mean()
            _y = _y[x].mean()
            _z = _z[x].mean()
            return np.array([_x,_y,_z])

        features       = []
        labels         = []
        for d in data:
            image_t1   = itk.GetImageFromArray(d[0][0][0])
            mask_t1    = itk.GetImageFromArray((d[0][0][4]!=0).astype(np.int))
            image_t1ce = itk.GetImageFromArray(d[0][0][1])
            mask_t1ce  = itk.GetImageFromArray((d[0][0][4]!=0).astype(np.int))
            image_t2   = itk.GetImageFromArray(d[0][0][2])
            mask_t2    = itk.GetImageFromArray((d[0][0][4]!=0).astype(np.int))
            image_flair= itk.GetImageFromArray(d[0][0][3])
            mask_flair = itk.GetImageFromArray((d[0][0][4]!=0).astype(np.int))
            age   = d[0][1]

            feature = [age]
            # for name in ['firstorder','shape','glcm','glrlm','glszm','ngtdm','gldm']:
            #     #'firstorder','shape','glcm','glrlm','glszm','ngtdm','gldm'
            #     obj = globals()[name]
            #     attr = [i for i in dir(obj) if i.startswith('Radiomics')][0]
            #     feature.extend(list(getattr(obj,attr)(image_t1,mask_t1).execute().values()))
            
            # for name in ['firstorder','shape','glcm','glrlm','glszm','ngtdm','gldm']:
            #     #'firstorder','shape','glcm','glrlm','glszm','ngtdm','gldm'
            #     obj = globals()[name]
            #     attr = [i for i in dir(obj) if i.startswith('Radiomics')][0]   
            #     feature.extend(list(getattr(obj,attr)(image_t1ce,mask_t1ce).execute().values()))

            for name in ['firstorder','shape','glcm','glrlm','glszm','ngtdm','gldm']:
                #'firstorder','shape','glcm','glrlm','glszm','ngtdm','gldm'
                obj = globals()[name]
                attr = [i for i in dir(obj) if i.startswith('Radiomics')][0]  
                feature.extend(list(getattr(obj,attr)(image_t2,mask_t2).execute().values()))

            # for name in ['firstorder','shape','glcm','glrlm','glszm','ngtdm','gldm']:
            #     #'firstorder','shape','glcm','glrlm','glszm','ngtdm','gldm'
            #     obj = globals()[name]
            #     attr = [i for i in dir(obj) if i.startswith('Radiomics')][0]  
            #     feature.extend(list(getattr(obj,attr)(image_flair,mask_flair).execute().values()))

            feature = np.array(feature)

            if len(d) > 1:
                label = d[1]
            else:
                label = None

            features.append(feature)

            brain = d[0][0][0] != 0
            seg   = d[0][0][4]
            age   = d[0][1]
            if len(d) > 1:
                label = d[1]
            else:
                label = None
            
            brain_center = center(brain)
            brain_volume = brain.sum()

            def ratio(x):
                return np.array([x.sum() / brain_volume])
            def offset(x):
                if(x.sum() == 0):
                    return brain_center
                return center(x) - brain_center
            def distance(x):
                return np.array([np.sqrt((offset(x)**2).sum())])
            def surface_ratio(x):
                if x.sum() == 0:
                    return np.array([0])
                kernel = torch.zeros(3,3,3,3)
                kernel[0][1,1,0] = -1
                kernel[0][1,1,2] = 1 
                kernel[1][1,0,1] = -1
                kernel[1][1,2,1] = 1
                kernel[2][0,1,1] = -1
                kernel[2][2,1,1] = 1                
                t = F.conv3d(torch.tensor(x[None,None,:,:,:]).float(),kernel[:,None,:,:,:],padding=1)[0].any(0).numpy()
                surface = t.sum()
                volume = x.sum()
                return np.array([surface/volume],dtype=np.float32)

            necrosis = seg==1 
            active   = (seg==1)|(seg==4)
            edema    = (seg==1)|(seg==2)|(seg==4)

            new_feature = []

            for func in [ratio,offset,distance,surface_ratio]:
                for mask in [necrosis,edema,active]:
                    new_feature.append(func(mask))
            # for func in [ratio,offset]:
            #     for mask in [edema,active]:
            #         feature.append(func(mask))
            features.append(np.concatenate(new_feature)) 
            features = np.concatenate(features)
                              
            if label is not None:
                labels.append(label)
            
        dic = [0, 1, 11, 96, 97, 108, 110, 112, 113, 115, 116, 118, 119, 120, 121, 122]
        features_sel = []
        for i in dic:
            features_sel.append(features[i])
        features_sel = np.atleast_2d(features_sel)
        print(features_sel.shape)

        if len(labels) > 0:
            labels   = np.array(labels)
            if self.return_type == 'tensor':
                features = torch.tensor(features_sel)
                labels   = torch.tensor(labels)
            return features_sel,labels
        else:
            if self.return_type == 'tensor':
                features_sel = torch.tensor(features_sel)
            return features_sel

def give_me_all(loader):
    path = os.path.join(loader.dataset.root_dir,'..','.cache','dataset','all',loader.dataset.name+'.pkl')

    if os.path.exists(path):
        # load
        print('Load Directly')
        data = pkl.load(open(path,'rb'))
    else:
        # process
        if loader.dataset.name=='train' :
            features_sel,labels = [],[]
            for i in tqdm(loader.dataset,desc="Preprocessing"):
                feature,label = loader.collate_fn([i])
                features_sel.append(feature)
                labels.append(label)
            cat_func = {
                np.ndarray:np.concatenate,
                torch.Tensor:torch.cat
            }[type(feature)]
            features_sel = cat_func(features_sel)
            labels   = cat_func(labels)
            data = (features_sel,labels)
        elif loader.dataset.name == 'test':
            features_sel= []
            for i in tqdm(loader.dataset,desc="Preprocessing"):
                feature = loader.collate_fn([i])
                features_sel.append(feature)
            cat_func = {
                np.ndarray:np.concatenate,
                torch.Tensor:torch.cat
            }[type(feature)]
            features_sel = cat_func(features_sel)
            data = features_sel
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
    dataset = SurCombine(ftype=['t1','t1ce','t2','flair','seg'])
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