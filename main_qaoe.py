
from lib import *
from dataset import Dataset_Base
from model import VIOLET_Base
from agent import Agent_Base

class Dataset_QAOE(Dataset_Base):
    def __init__(self, args, split):
        super().__init__(args)
        
        self.img = pickle.load(open('./_data/img_%s.pkl'%(self.args['dataset']), 'rb'))
        self.txt = json.load(open('./_data/txt_%s.json'%(self.args['task']), 'r'))[split]
        
    def __len__(self):
        return len(self.txt)
    
    def __getitem__(self, idx):
        item = self.txt[idx]
        
        img = []
        for b in self.img[item['video']]:
            img.append(self.str2img(b).unsqueeze(0))
        img = T.cat(img, dim=0)
        
        txt, mask = self.str2txt(item['question'])
        
        return img, txt, mask, item['answer']
    
class VIOLET_QAOE(VIOLET_Base):
    def __init__(self, size_vocab):
        super().__init__()
        
        self.fc = T.nn.Sequential(*[T.nn.Dropout(0.1), 
                                    T.nn.Linear(768, 768*2), T.nn.ReLU(inplace=True), 
                                    T.nn.Linear(768*2, size_vocab)])
    
    def forward(self, img, txt, mask):
        (_B, _T, _, _H, _W), (_, _X) = img.shape, txt.shape
        _h, _w = _H//32, _W//32
        
        feat_img, mask_img, feat_txt, mask_txt = self.go_feat(img, txt, mask)
        out, _ = self.go_cross(feat_img, mask_img, feat_txt, mask_txt)
        out = self.fc(out[:, (1+_h*_w)*_T, :])
        
        return out

class Agent_QAOE(Agent_Base):
    def __init__(self, args, model):
        super().__init__(args, model)
    
    def step(self, img, txt, mask, ans, is_train):
        self.optzr.zero_grad()
        with T.cuda.amp.autocast():
            out = self.model(img.cuda(), txt.cuda(), mask.cuda())
            ls = self.loss_func(out, ans.cuda())
        if is_train==True:
            self.scaler.scale(ls).backward()
            self.scaler.step(self.optzr)
            self.scaler.update()
            return ls.item()
        else:
            out = T.argmax(out, dim=1)
            ac = (out==ans.cuda()).float().mean().item()
            return ac
    
    def go_dl(self, dl, is_train):
        ret = []
        for img, txt, mask, ans in tqdm(dl, ascii=True):
            ret.append(self.step(img, txt, mask, ans, is_train))
        ret = float(np.average(ret))
        
        return ret    
    
if __name__=='__main__':
    args = json.load(open(sys.argv[1], 'r'))
    args['size_batch'] = args['size_batch']*T.cuda.device_count()
    args['path_output'] = '_snapshot/_%s_%s'%(args['task'], datetime.now().strftime('%Y%m%d%H%M%S'))
    os.makedirs(args['path_output'], exist_ok=True)
    json.dump(args, open('%s/args.json'%(args['path_output']), 'w'), indent=2)
    print(args)
    
    dl_tr, dl_vl, dl_ts = [T.utils.data.DataLoader(Dataset_QAOE(args, split), 
                                                   batch_size=args['size_batch'], shuffle=(split=='train'), 
                                                   num_workers=32, pin_memory=True)\
                           for split in ['train', 'val', 'test']]
    
    log = {'ls_tr': [], 'ac_vl': [], 'ac_ts': []}
    json.dump(log, open('%s/log.json'%(args['path_output']), 'w'), indent=2)
    
    model = T.nn.DataParallel(VIOLET_QAOE(args['size_vocab']).cuda())
    model.module.load_ckpt(args['path_ckpt'])
    T.save(model.module.state_dict(), '%s/ckpt_violet_%s_0.pt'%(args['path_output'], args['task']))
    
    agent = Agent_QAOE(args, model)
    for e in tqdm(range(args['size_epoch']), ascii=True):
        model.train()
        ls_tr = agent.go_dl(dl_tr, True)
        
        model.eval()
        ac_vl = agent.go_dl(dl_vl, False)
        ac_ts = agent.go_dl(dl_ts, False)
        
        log['ls_tr'].append(ls_tr), log['ac_vl'].append(ac_vl), log['ac_ts'].append(ac_ts)
        json.dump(log, open('%s/log.json'%(args['path_output']), 'w'), indent=2)
        T.save(model.module.state_dict(), '%s/ckpt_violet_%s_%d.pt'%(args['path_output'], args['task'], e+1))
        print('Ep %d: %.6f %.6f %.6f'%(e+1, ls_tr, ac_vl, ac_ts))
        