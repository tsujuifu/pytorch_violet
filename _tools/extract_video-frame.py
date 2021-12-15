
import argparse, av, base64, io, pickle

from glob import glob
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path', required=True, type=str)
    parser.add_argument('--sample', required=True, type=int)
    
    args = parser.parse_args()
    
    return args

if __name__=='__main__':
    args = get_args()
    
    lst = glob('_%s/*.mp4'%(args.path))
    
    pkl = {}
    for f in tqdm(lst, ascii=True):
        vid = f.split('/')[-1].replace('.mp4', '')
        
        imgs = []
        for pack in av.open(f).demux():
            for buf in pack.decode():
                if str(type(buf))=="<class 'av.video.frame.VideoFrame'>":
                    imgs.append(buf.to_image().convert('RGB'))
        N = len(imgs)/(args.sample+1)
        
        pkl[vid] = []
        for i in range(args.sample):
            buf = io.BytesIO()
            imgs[int(N*(i+1))].save(buf, format='JPEG')
            pkl[vid].append(str(base64.b64encode(buf.getvalue()))[2:-1])
    pickle.dump(pkl, open('%s.pkl'%(args.path), 'wb'))
    
