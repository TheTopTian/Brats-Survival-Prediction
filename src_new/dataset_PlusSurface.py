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

        def center(x):
            _x,_y,_z = np.meshgrid(*[np.arange(i) for i in x.shape])
            _x = _x[x].mean()
            _y = _y[x].mean()
            _z = _z[x].mean()
            return np.array([_x,_y,_z])

        features       = []
        labels         = []
        for d in data:
            brain = d[0][0][0] != 0
            seg   = d[0][0][1]
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

            feature = [np.array([age])]
            for func in [ratio,offset,distance,surface_ratio]:
                for mask in [necrosis,active]:
                    feature.append(func(mask))
            for func in [ratio]:
                for mask in [edema]:
                    feature.append(func(mask))
            features.append(np.concatenate(feature))  
                              
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

    '''
    run.py
        K = 10
        model = LeNet3D()
        optimizer = torch.optim.Adam(model.parameters(),....)
        total_metirc = []
        for k in range(K):
            # reset dataset everry cross valid iteration
            dataset = BrainMRISuvival(k=k,K=K)
            train_loader = dataset.loader('train')
            valid_loader = dataset.loader('valid')

            # reset model every cross valid iteration
            model.reset_parameters()
            
            # training loop 
            for ep in range(epoch):
                model.train() 
                for (x1,x2),y in train_loader:
                    p = model(x1,x2)
                    l = loss(p,y)
                    optimizer.zero_grad()
                    l.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    l_m = []
                    for (x1,x2),y in valid_loader:
                        p = model(x1,x2)
                        m = metric(p,y)
                        l_m.append(m)
                    m = np.array(l_m)
                print(f"Epoch:{ep} metric:{metric.mean()}({metric.std()})")

            model.save(affix=f'[k{k}]')
    '''