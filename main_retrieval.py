
from lib import *
from dataset import Dataset_Base
from model import VIOLET_Base
from agent import Agent_Base

class Dataset_Retrieval(Dataset_Base):
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
        
        txt, mask = self.str2txt(item['caption'])
        
        return img, txt, mask

class VIOLET_Retrieval(VIOLET_Base):
    def __init__(self):
        super().__init__()
        
        self.fc = T.nn.Sequential(*[T.nn.Dropout(0.1), 
                                    T.nn.Linear(768, 768*2), T.nn.ReLU(inplace=True), 
                                    T.nn.Linear(768*2, 1)])
    
    def forward(self, img, txt, mask):
        (_B, _T, _, _H, _W), (_, _X) = img.shape, txt.shape
        _h, _w = _H//32, _W//32
        
        feat_img, mask_img, feat_txt, mask_txt = self.go_feat(img, txt, mask)
        
        pdt_feat_img, pdt_mask_img, pdt_feat_txt, pdt_mask_txt = [], [], [], []
        for i in range(_B):
            for j in range(_B):
                pdt_feat_img.append(feat_img[i].unsqueeze(0)), pdt_mask_img.append(mask_img[i].unsqueeze(0))
                pdt_feat_txt.append(feat_txt[j].unsqueeze(0)), pdt_mask_txt.append(mask_txt[j].unsqueeze(0))
        pdt_feat_img, pdt_mask_img, pdt_feat_txt, pdt_mask_txt = [T.cat(x, dim=0) for x in [pdt_feat_img, pdt_mask_img, 
                                                                                            pdt_feat_txt, pdt_mask_txt]]
        out, _ = self.go_cross(pdt_feat_img, pdt_mask_img, pdt_feat_txt, pdt_mask_txt)
        out = self.fc(out[:, (1+_h*_w)*_T, :]).squeeze().view([_B, _B]) / 0.05
        
        ans = T.tensor([i for i in range(_B)]).long().cuda()
        
        return out, ans

class Agent_Retrieval(Agent_Base):
    def __init__(self, args, model):
        super().__init__(args, model)
    
    def step(self, img, txt, mask, is_train):
        self.optzr.zero_grad()
        with T.cuda.amp.autocast():
            out, ans = self.model(img.cuda(), txt.cuda(), mask.cuda())
            ls = self.loss_func(out, ans)
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
        for img, txt, mask in tqdm(dl, ascii=True):
            ret.append(self.step(img, txt, mask, is_train))
        ret = float(np.average(ret))
        
        return ret    
    
if __name__=='__main__':
    args = json.load(open(sys.argv[1], 'r'))
    args['size_batch'] = args['size_batch']*T.cuda.device_count()
    args['path_output'] = '_snapshot/_%s_%s'%(args['task'], datetime.now().strftime('%Y%m%d%H%M%S'))
    os.makedirs(args['path_output'], exist_ok=True)
    json.dump(args, open('%s/args.json'%(args['path_output']), 'w'), indent=2)
    print(args)
    
    dl_tr, dl_vl, dl_ts = [T.utils.data.DataLoader(Dataset_Retrieval(args, split), 
                                                   batch_size=args['size_batch'], shuffle=(split=='train'), 
                                                   num_workers=32, pin_memory=True)\
                           for split in ['train', 'val', 'test']]
    
    log = {'ls_tr': [], 'ac_vl': [], 'ac_ts': []}
    json.dump(log, open('%s/log.json'%(args['path_output']), 'w'), indent=2)
    
    model = T.nn.DataParallel(VIOLET_Retrieval().cuda())
    model.module.load_ckpt(args['path_ckpt'])
    T.save(model.module.state_dict(), '%s/ckpt_violet_%s_0.pt'%(args['path_output'], args['task']))
    
    agent = Agent_Retrieval(args, model)
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
        