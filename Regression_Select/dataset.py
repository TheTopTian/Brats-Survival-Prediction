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

DATASET_PATH = '..'
TRAIN_PATH = 'MICCAI_BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData'
VALID_PATH = 'MICCAI_BraTS2020_ValidationData\\MICCAI_BraTS2020_ValidationData'

# # Read the CSV file
# path = os.path.join(DATASET_PATH,TRAIN_PATH)
# df = pd.read_csv(os.path.join(path,'survival_info.csv'))
# df = df[df.Extent_of_Resection.isin(['GTR'])] #只保留GTR的行数

# titles = np.array(df['Brats20ID'])
# # titles = np.array(df.iloc[:,0])
# ages = np.array(df.iloc[:,1])
# survival_dates = np.array(df.iloc[:,2]).tolist()

class SegTaskRead(Dataset):
    def __init__(self,root_dir,k=0,K=5,name='train'):
        '''
            basic config
        '''
        self.name = name
        self.path = {
            'train':os.path.join(root_dir,DATASET_PATH,TRAIN_PATH),
            'valid':os.path.join(root_dir,DATASET_PATH,TRAIN_PATH),
            'test':os.path.join(root_dir,DATASET_PATH,VALID_PATH)
        }[name]
        filenames = list(filter(lambda x:os.path.isdir(x),map(lambda x:os.path.join(self.path,x),os.listdir(self.path))))
        block_size = int(np.ceil(len(filenames)/K))
        self.filenames = []
        if self.name == 'train':
            for i in range(K):
                if i==k:
                    pass 
                else:
                    self.filenames.extend(filenames[i*block_size:(i+1)*block_size])
        elif self.name == 'valid':
            self.filenames = filenames[k*block_size:(k+1)*block_size]
        else:
            self.filenames = filenames
        self._init_feat()
        self._init_label()

    def _init_feat(self): 
        '''
            read the number of features and shape
        '''
        index = 1
        filename = [filename for filename in os.listdir(self.filenames[index]) if 't1' in filename][0]
        image    = nib.load(os.path.join(self.filenames[index],filename)).get_fdata()
        shape    = image.shape

        self.shape = shape
        self.n_feats = 4

    def _init_label(self):
        '''
            read the number of classes
        '''
        index = 1
        if self.name == 'train':
            filename = [filename for filename in os.listdir(self.filenames[index]) if 'seg'  in filename][0]
            image    = nib.load(os.path.join(self.filenames[index],filename)).get_fdata()
            self.n_classes = len(np.bincount(np.unique(image).astype(int)))

    # def interpolate(self,x:torch.Tensor):
    #     assert isinstance(x,torch.Tensor)
    #     shape  = x.shape[2:]
    #     shape  = [int(i*self.scale) for i in shape]
    #     x = F.interpolate(x,size=shape,mode='trilinear')
    #     return x
    def __len__(self):
        return len(self.filenames)
    
    def _read_feat(self,index):
        path = self.filenames[index]
        filenames = os.listdir(path)
        image_filename_t1 = [filename for filename in filenames if 't1'  in filename][0]
        image_filename_t2 = [filename for filename in filenames if 't2'  in filename][0]
        image_filename_t1ce = [filename for filename in filenames if 't1ce'  in filename][0]
        image_filename_flair = [filename for filename in filenames if 'flair'  in filename][0]
        image_t1 = nib.load(os.path.join(path,image_filename_t1)).get_fdata()
        image_t2 = nib.load(os.path.join(path,image_filename_t2)).get_fdata()
        image_t1ce = nib.load(os.path.join(path,image_filename_t1ce)).get_fdata()
        image_flair = nib.load(os.path.join(path,image_filename_flair)).get_fdata()
        return image_t1,image_t2,image_t1ce,image_flair 
    def _read_label(self,index):
        path = self.filenames[index]
        filenames = os.listdir(path)
        label_filename = [filename for filename in filenames if 'seg' in filename][0]
        label = nib.load(os.path.join(path,label_filename)).get_fdata()
        return label
    def __getitem__(self,index):
        feat = self._read_feat(index)
        if self.name != 'test':
            label = self._read_label(index)
            return feat,label
        else:
            return feat

class SegCombine:
    def __init__(self,root_dir='.',k=0,K=5,SubDataset=SegTaskRead):
        self.root_dir = root_dir

        self.train = SubDataset(root_dir,k,K,'train')
        self.valid = SubDataset(root_dir,k,K,'valid')

        assert self.train.shape    == self.valid.shape 
        assert self.train.n_feats  == self.valid.n_feats
        # assert self.train.n_classes== self.valid.n_classes
        self.shape = self.train.shape 
        self.n_feats   = self.train.n_feats
        self.n_classes = self.train.n_classes

    def __len__(self):
        return len(self.train) + len(self.valid)

    def collate_fn(self,data):
        # images_t1 = []
        # images_t2 = []
        # images_t1ce = []
        # images_flair = []
        feats  = []
        labels = []
        for d in data:
            '''
            data:
                [
                    (
                        (
                        np.ndarray[240,240,155],
                        np.ndarray[240,240,155],
                        np.ndarray[240,240,155],
                        np.ndarray[240,240,155]
                        ),
                        np.ndarray[240,240,155]
                    ),
                    (
                        (
                        np.ndarray[240,240,155],
                        np.ndarray[240,240,155],
                        np.ndarray[240,240,155],
                        np.ndarray[240,240,155]
                        ),
                        np.ndarray[240,240,155]
                    ),
                    ...
                ]
                ==stack==>
                feats:[
                    np.ndarray[4,240,240,155],
                    np.ndarray[4,240,240,155],
                    ...
                ]
                labels:[
                    np.ndarray[240,240,155],
                    np.ndarray[240,240,155],
                    ...
                ]
            '''
            feats.append(np.stack(d[0],0))
            # images_t1.append(d[0])
            # images_t2.append(d[0])
            # images_t1ce.append(d[0])
            # images_flair.append(d[0])
            labels.append(d[1])
        '''
            feats:[
                np.ndarray[4,240,240,155],
                np.ndarray[4,240,240,155],
                ...
            ]
            labels:[
                np.ndarray[240,240,155],
                np.ndarray[240,240,155],
                ...
            ]
            ==stack==>
            feats:np.ndarray[batch,4,240,240,155]
            labels:np.ndarray[batch,240,240,155]
        '''
        feats = torch.tensor(np.stack(feats,0)).float()
        # image_t1 = torch.tensor(np.stack(images_t1,0)).float()
        # image_t2 = torch.tensor(np.stack(images_t2,0)).float()
        # image_t1ce = torch.tensor(np.stack(images_t1ce,0)).float()
        # image_flair = torch.tensor(np.stack(images_flair,0)).float()

        label = torch.tensor(np.stack(labels,0)).long()
        # image_t1 = image_t1[:,None,...] 
        # image_t2 = image_t2[:,None,...]
        # image_t1ce = image_t1ce[:,None,...]
        # image_flair = image_flair[:,None,...]
        # image = torch.cat((image_t1,image_t2,image_t1ce,image_flair),1)

        # shape  = image.shape[2:]

        # image = F.interpolate(image,size=shape,mode='trilinear')
        # label = F.interpolate(label[:,None,...],size=shape,mode='nearest')[:,0,...].long()
        # return image,label
        return feats,label
    
    def loader(self,name='train',batch_size=1,shuffle=False):
        return DataLoader(
            getattr(self,name),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn
        )

class SurTaskRead(SegTaskRead):
    def __init__(self,root_dir,k,K,name='train',ftype=None,threshold=None):
        '''
            assertion
        '''
        if ftype is None:
            ftype = ['t1','t2','t1ce','flair','seg']
        assert all([i in ['t1','t2','t1ce','flair','seg'] for i in ftype]),f'ftype should be one of t1,t2,t1ce,flair,seg'
      
        '''
            basic config
        '''
        self.root_dir  = root_dir
        self.threshold = np.array(threshold) if threshold is not None else None
        self.ftype     = ftype
        self.name      = name
        self.path = {
            'train':os.path.join(root_dir,DATASET_PATH,TRAIN_PATH),
            'valid':os.path.join(root_dir,DATASET_PATH,TRAIN_PATH),
            'test':os.path.join(root_dir,DATASET_PATH,VALID_PATH)
        }[name]
        

        '''
            read survival days
        '''
        if name in ['train','valid']:
            df = pd.read_csv(os.path.join(self.path,'survival_info.csv'))
            df = df[df['Extent_of_Resection']=='GTR']
            survival_days = df['Survival_days'].tolist()
            selected1     = ~np.array(['ALIVE' in i for i in survival_days])
            ids           = np.array(df['Brats20ID'])[selected1]
            
            block_size = int(np.ceil(len(ids)/K))
            indexes = np.arange(len(ids))
            selected2 = []
            if name == 'train':
                for i in range(K):
                    if k==i:
                        pass 
                    else:
                        selected2.extend(indexes[i*block_size:(i+1)*block_size])
            elif name == 'valid':
                selected2 = indexes[k*block_size:(k+1)*block_size]
            else:
                selected2 = indexes
            
            self.survival_days = np.array(df['Survival_days'])[selected1].astype(np.float32)[selected2]
            self.ages = np.array(df['Age'])[selected1].astype(np.float32)[selected2]
            self.filenames = np.array([os.path.join(self.path,filename) for filename in ids])[selected2]
        else:
            df = pd.read_csv(os.path.join(self.path,'survival_evaluation.csv'))
            ids = df['BraTS20ID'].tolist()
            ages = np.array(df['Age']).astype(np.float32)
            self.ages = ages 
            self.filenames = np.array([os.path.join(self.path,filename) for filename in ids])

        if self.threshold is not None and name != 'test':
            self.bucket = np.zeros_like(self.survival_days)
            MIN = self.survival_days.min()
            MAX = self.survival_days.max()
            self.survival_days_max = MAX 
            self.survival_days_min = MIN
            assert (MIN <= self.threshold).all() and (self.threshold < MAX).all(),f"threshold {threshold} is out of range of [{MIN},{MAX}) "
            for i,high in enumerate(threshold+[MAX]):
                # [low,high)
                if i==0:
                    low = MIN 
                else:
                    low = self.threshold[i-1]
                mask = (low <= self.survival_days) & (self.survival_days < high)
                self.bucket[mask] = i 
        '''
            read grade
        '''
        # df       = pd.read_csv(os.path.join(self.path,'name_mapping.csv'))
        # id2grade = dict(zip(df['BraTS_2020_subject_ID'],df['Grade']))
        # grades   = [id2grade[x] for x in ids]

        

        self._init_feat()
        if self.name != 'test':
            self._init_label()

    def _init_feat(self): 
        '''
            read the number of features and shape
        '''
        index = 1
        filename = [filename for filename in os.listdir(self.filenames[index]) if self.ftype[0] in filename][0]
        image    = nib.load(os.path.join(self.filenames[index],filename)).get_fdata()
        shape    = image.shape

        self.shape = shape
        self.n_feats = len(self.ftype)

    def _init_label(self):
        if self.threshold is None:
            self.n_classes = None
        else:
            self.n_classes = len(self.threshold) + 1

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
        if self.threshold is None:
            return self.survival_days[index]    
        else:
            return self.bucket[index]
    def __getitem__(self,index):
        if self.name == 'test':
            assert ('seg' not in self.ftype), "Test Dataset doesn't have segmentation files"
        return super().__getitem__(index)

class SurCombine(SegCombine):
    def __init__(self,k=0,K=5,root_dir='.',ftype=['t1','t2','t1ce','flair','seg'],threshold=[10*30,15*30]):
        if threshold is not None:
            threshold.sort()
        self.threshold = threshold
        self.root_dir = root_dir

        self.train = SurTaskRead(root_dir,k,K,'train',ftype,threshold)
        self.valid = SurTaskRead(root_dir,k,K,'valid',ftype,threshold)
        self.test  = SurTaskRead(root_dir,k,K,'test',ftype,threshold)

        assert self.train.shape    == self.valid.shape 
        assert self.train.n_feats  == self.valid.n_feats
        # assert self.train.n_classes== self.valid.n_classes
        self.shape = self.train.shape 
        self.n_feats   = self.train.n_feats
        self.n_classes = self.train.n_classes
        if threshold is not None:
            self.survial_days_max = max(self.train.survival_days_max,self.valid.survival_days_max)
            self.survial_days_min = min(self.train.survival_days_min,self.valid.survival_days_min)
    def label_semantic(self,label):
        if isinstance(label,(list,tuple)):
            label = np.array(label)
        elif isinstance(label,np.ndarray):
            pass 
        elif isinstance(label,torch.Tensor):
            label = label.detach().cpu().numpy()
        else:
            raise Exception(f'label type not support {type(label)}')
        
        assert len(label.shape) == 1, f'only process 1-d label, but got shape {label.shape}'

        MIN = self.survial_days_min
        MAX = self.survial_days_max    
        low = np.array([MIN] + self.threshold)
        high = np.array(self.threshold + [MAX])
        lows = low[label]
        highs = high[label]
        return lows,highs
    def collate_fn(self,data):
        feats_mri = []
        feats_age = []
        labels = []
        for d in data:
            feats_mri.append(np.stack(d[0][0],0))
            feats_age.append(d[0][1])
            if len(d) > 1:
                labels.append(d[1])
        feats_mri = torch.tensor(np.stack(feats_mri,0)).float()
        feats_age = torch.tensor(feats_age).float()
        if len(labels) > 0:
            if self.threshold is None:
                labels = torch.tensor(labels).float()
            else:
                labels = torch.tensor(labels).long()
            return (feats_mri,feats_age),labels
        else:
            return (feats_mri,feats_age)


class SurCombineFeaturing(SurCombine):
    def __init__(self,k=0,K=5,root_dir='.',return_type = 'array'):
        assert return_type in ['tensor','array']
        self.return_type = return_type
        super().__init__(k,K,root_dir,['t1','t1ce','t2','flair','seg'],None)

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
            for name in ['firstorder','shape','glcm','glrlm','glszm','ngtdm','gldm']:
                #'firstorder','shape','glcm','glrlm','glszm'
                obj = globals()[name]
                attr = [i for i in dir(obj) if i.startswith('Radiomics')][0]
                feature.extend(list(getattr(obj,attr)(image_t1,mask_t1).execute().values()))
            
            # for name in ['firstorder','shape','glcm','glrlm','glszm','ngtdm','gldm']:
            #     #'firstorder','shape','glcm','glrlm','glszm'
            #     obj = globals()[name]
            #     attr = [i for i in dir(obj) if i.startswith('Radiomics')][0]   
            #     feature.extend(list(getattr(obj,attr)(image_t1ce,mask_t1ce).execute().values()))
            
            # for name in ['firstorder','shape','glcm','glrlm','glszm','ngtdm','gldm']:
            #     #'firstorder','shape','glcm','glrlm','glszm'
            #     obj = globals()[name]
            #     attr = [i for i in dir(obj) if i.startswith('Radiomics')][0]  
            #     feature.extend(list(getattr(obj,attr)(image_t2,mask_t2).execute().values()))

            # for name in ['firstorder','shape','glcm','glrlm','glszm','ngtdm','gldm']:
            #     #'firstorder','shape','glcm','glrlm','glszm'
            #     obj = globals()[name]
            #     attr = [i for i in dir(obj) if i.startswith('Radiomics')][0]  
            #     feature.extend(list(getattr(obj,attr)(image_flair,mask_flair).execute().values()))

            feature = np.array(feature)
            features.append(feature)

            brain = d[0][0][3] != 0
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
            
        # features = np.stack(features,0)
        features = np.atleast_2d(features)
        print(features.shape)

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
        features,labels = [],[]
        for i in tqdm(loader.dataset,desc="Preprocessing"):
            feature,label = loader.collate_fn([i])
            features.append(feature)
            labels.append(label)
        cat_func = {
            np.ndarray:np.concatenate,
            torch.Tensor:torch.cat
        }[type(feature)]
        stack_func = {
            np.ndarray:np.stack,
            torch.Tensor:torch.stack
        }[type(feature)]
        features = cat_func(features)
        labels   = cat_func(labels)
        data = (features,labels)

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
    print(len(dataset.valid))
    # print(next(iter(dataset.loader('train')))[0].shape)
    # print(next(iter(dataset.loader('train')))[1].shape)
    loader = dataset.loader('train',batch_size=2)
    (x1,x2),y = next(iter(loader))
    print('x1',x1.shape,x1.dtype)
    print('x2',x2.shape,x2.dtype)
    print('y',y.shape,y.dtype)


    dataset = SegCombine()
    print(dataset.__class__.__name__)
    loader = dataset.loader('train',batch_size=2)
    x,y = next(iter(loader))
    print('x',x.shape,x.dtype)
    print('y',y.shape,y.dtype)