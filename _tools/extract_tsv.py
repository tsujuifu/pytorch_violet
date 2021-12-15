
import argparse, pickle
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path', required=True, type=str)
    
    args = parser.parse_args()
    
    return args

if __name__=='__main__':
    args = get_args()
    
    pkl = pickle.load(open('%s.pkl'%(args.path), 'rb'))
    
    file_tsv, file_lineidx = open('%s.tsv'%(args.path), 'w'), open('%s.lineidx'%(args.path), 'w')
    for vid in tqdm(pkl, ascii=True):
        file_lineidx.write('%d\n'%(file_tsv.tell()))
        file_tsv.write(vid)
        for b in pkl[vid]:
            file_tsv.write('\t%s'%(b))
        file_tsv.write('\n')
        
        file_tsv.flush(), file_lineidx.flush()
        
