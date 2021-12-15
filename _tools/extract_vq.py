
import argparse, base64, io, pickle

from tqdm import tqdm

import numpy as np
import torch as T
import torchvision as TV
from dall_e import map_pixels, unmap_pixels, load_model

from PIL import Image

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path', required=True, type=str)
    parser.add_argument('--frame', required=True, type=int)
    
    args = parser.parse_args()
    
    return args

def proc_buf(buf, _F):
    img = Image.open(io.BytesIO(base64.b64decode(buf)))
    w, h = img.size
    img = TV.transforms.Compose([TV.transforms.Pad([0, (w-h)//2] if w>h else [(h-w)//2, 0]), 
                                 TV.transforms.Resize([_F, _F]), 
                                 TV.transforms.ToTensor()])(img).unsqueeze(0)
    img = map_pixels(img)
    return img

if __name__=='__main__':
    args = get_args()
    
    dalle_enc = load_model('encoder.pkl', T.device('cpu')).cuda() # https://cdn.openai.com/dall-e/encoder.pkl
    # dalle_dec = load_model('decoder.pkl', T.device('cpu')).cuda() # https://cdn.openai.com/dall-e/decoder.pkl
    
    pkl = pickle.load(open('%s.pkl'%(args.path), 'rb'))
    
    vq = {}
    for vid in tqdm(pkl, ascii=True):
        imgs = [proc_buf(b, int(args.frame//32*8)) for b in pkl[vid]]
        imgs = T.cat(imgs, dim=0)
        
        z = dalle_enc(imgs.cuda())
        z = T.argmax(z, dim=1)
        vq[vid] = z.data.cpu().numpy().astype(np.int16)
        
        '''o = T.nn.functional.one_hot(z, num_classes=dalle_enc.vocab_size).permute(0, 3, 1, 2).float()
        o = dalle_dec(o).float()
        rec = unmap_pixels(T.sigmoid(o[:, :3]))
        rec = [TV.transforms.ToPILImage(mode='RGB')(r) for r in rec]'''
    pickle.dump(vq, open('%s_vq.pkl'%(args.path), 'wb'))
    