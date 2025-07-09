from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2, os, json

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

class Dinov2Matcher:

  def __init__(self, repo_name="facebookresearch/dinov2", 
               model_name="dinov2_vitb14", smaller_edge_size=448, half_precision=False, device="cpu"):
    self.repo_name = repo_name
    self.model_name = model_name
    self.smaller_edge_size = smaller_edge_size
    self.half_precision = half_precision
    self.device = device

    if self.half_precision:
      self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).half().to(self.device)
    else:
      self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).to(self.device)

    self.model.eval()

    self.transform = transforms.Compose([
        transforms.Resize(size=smaller_edge_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # imagenet defaults
      ])

  # https://github.com/facebookresearch/dinov2/blob/255861375864acdd830f99fdae3d9db65623dafe/notebooks/features.ipynb
  def prepare_image(self, rgb_image_numpy):
    image = Image.fromarray(rgb_image_numpy)
    image_tensor = self.transform(image)
    resize_scale = image.width / image_tensor.shape[2]

    # Crop image to dimensions that are a multiple of the patch size
    height, width = image_tensor.shape[1:] # C x H x W
    cropped_width, cropped_height = width - width % self.model.patch_size, height - height % self.model.patch_size # crop a bit from right and bottom parts
    image_tensor = image_tensor[:, :cropped_height, :cropped_width]

    grid_size = (cropped_height // self.model.patch_size, cropped_width // self.model.patch_size)
    return image_tensor, grid_size, resize_scale
  
  def prepare_mask(self, mask_image_numpy, grid_size, resize_scale):
    cropped_mask_image_numpy = mask_image_numpy[:int(grid_size[0]*self.model.patch_size*resize_scale), :int(grid_size[1]*self.model.patch_size*resize_scale)]
    image = Image.fromarray(cropped_mask_image_numpy)
    resized_mask = image.resize((grid_size[1], grid_size[0]), resample=Image.Resampling.NEAREST)
    resized_mask = np.asarray(resized_mask).flatten()
    return resized_mask
  
  def extract_features(self, image_tensor):
    with torch.inference_mode():
      if self.half_precision:
        image_batch = image_tensor.unsqueeze(0).half().to(self.device)
      else:
        image_batch = image_tensor.unsqueeze(0).to(self.device)

      tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()
    return tokens.cpu().numpy()
  
  def idx_to_source_position(self, idx, grid_size, resize_scale):
    row = (idx // grid_size[1])*self.model.patch_size*resize_scale + self.model.patch_size / 2
    col = (idx % grid_size[1])*self.model.patch_size*resize_scale + self.model.patch_size / 2
    return row, col
  
  def get_embedding_visualization(self, tokens, grid_size, resized_mask=None):
    pca = PCA(n_components=3)
    if resized_mask is not None:
      tokens = tokens[resized_mask]
    reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
    if resized_mask is not None:
      tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
      tmp_tokens[resized_mask] = reduced_tokens
      reduced_tokens = tmp_tokens
    reduced_tokens = reduced_tokens.reshape((*grid_size, -1))
    normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))
    return normalized_tokens

def procImage(ffil, forceLoad=False, cropW=False, saveNpz=True):
    if type(ffil) is str:
        image1 = cv2.cvtColor(cv2.imread(ffil, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if cropW:
            H,W = image1.shape[:2]
            image1 = image1[:,W//2:]
        npyFile= (ffil+'_feats.npz')
    else:
        image1 = ffil
        npyFile = None
        
    if (npyFile is not None) and os.path.isfile(npyFile) and (not forceLoad):
        d = np.load(npyFile)
        features1 = d['features']
        grid_size1 = d['grid_size']
        resize_scale1 = d['resize_scale']
        image = d['image']
        patch_size1 = d['patch_size']
    else:
        image_tensor1, grid_size1, resize_scale1 = dm.prepare_image(image1)
        features1 = dm.extract_features(image_tensor1)
        image = cv2.resize(image1,(image_tensor1.shape[2],image_tensor1.shape[1]))
        patch_size1 = dm.model.patch_size
        if saveNpz:
            np.savez(npyFile, image=image, grid_size=grid_size1, 
                     resize_scale=resize_scale1, features=features1,
                     patch_size=patch_size1)
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(features1)
    info ='size=%dx%d, grids=%dx%d, feats=%d'%(image.shape[0],image.shape[1], grid_size1[0], grid_size1[1], len(features1))
    
    return {'grid_size':grid_size1, 
            'image': image,
            'patch_size': patch_size1,
            'features': features1, 
            'resize_scale':resize_scale1, 
            'knn':knn,
            'info': info
           }

def xy2Index(x, y, obj):
    row = int(y//obj['patch_size'])
    col = int(x//obj['patch_size'])
    xGrids = obj['grid_size'][1]
    idx = row*xGrids + col
    return idx

def index2xy(idx, obj):
    psz = obj['patch_size']
    gsz = obj['grid_size']
    
    row = idx//gsz[1]
    col = idx%gsz[1]
    
    y = row*psz + psz*0.5
    x = col*psz + psz*0.5
    
    return [x,y]

def loadSourceFeats(refImageFile):
    obj = procImage(refImageFile)
    maskFile = refImageFile.replace('.png','.mask.0.png')
    mask = cv2.imread(maskFile)
    mask = np.max(mask, axis=2)
    maskScale = np.array(obj['image'].shape[:2])/np.array(mask.shape[:2])

    # x1,y1,x2,y2 = getBBox(ent['mobile']['bbox'])
    psz = obj['patch_size']
    xGrids = obj['grid_size'][1]

    rows, cols = np.where(mask>0)
    cols = cols*maskScale[0]
    rows = rows*maskScale[1]
    rc = np.c_[rows, cols]
    rc = (rc/psz).astype(int)
    rc = np.unique(rc, axis=0)

    srcFeatList = []
    for n in range(len(rc)):
        irow, icol = rc[n,:]
        idx = irow*xGrids + icol
        feat = obj['features'][idx:idx+1]
        srcFeatList.append(feat)
        
    return srcFeatList

dm = None
refDataset = None
feats = {}
def init(refJson):
    global dm, feats, refDataset
    
    srcDir = os.path.dirname(refJson)
    print('loading DINO model ...')
    dm = Dinov2Matcher(half_precision=False)
    
    print('loading reference dataset ...')
    refDataset = json.load(open(refJson,'r'))
    for typ in refDataset:
        srcFeatList = []
        for ent in refDataset[typ]['positives']:
            lst = loadSourceFeats(os.path.join(srcDir,ent))
            srcFeatList += lst
        srcFeatList = np.concatenate(srcFeatList, axis=0)
        print(typ, srcFeatList.shape)
        feats[typ] = srcFeatList
        
    print('DONE.')