
from lib import *
from dataset import Dataset_Base
from model import VIOLET_Base
from agent import Agent_Base

class Dist:
    def __init__(self):
        super().__init__()
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--local_rank", type=int)
        args = parser.parse_args()
        self.rank_local = args.local_rank
        
        T.cuda.set_device(self.rank_local)
        DIST.init_process_group(backend='nccl')
        
def iter_tqdm(item):
    return tqdm(item, ascii=True) if DIST.get_rank()==0 else item
        
class Dataset_Pretrain(Dataset_Base):
    def __init__(self, args, dataset, split, part=None):
        super().__init__(args)
        
        self.dataset, self.split, self.part = dataset, split, part
        
        self.txt = json.load(open('./_data/txt_%s.json'%(self.dataset), 'r'))[self.split]
        self.vq = pickle.load(open('./_data/vq_%s.pkl'%(self.dataset), 'rb'))
        self.lineidx = [int(p) for p in open('./_data/img_%s_train_%d.lineidx'%(self.dataset, self.part) if self.split=='train' \
                                             else './_data/img_%s_val.lineidx'%(self.dataset), 'r')]
    
    def read_tsv(self, worker_id):
        self.tsv = open('./_data/img_%s_train_%d.tsv'%(self.dataset, self.part) if self.split=='train' \
                        else './_data/img_%s_val.tsv'%(self.dataset), 'r')
    
    def __len__(self):
        return len(self.lineidx)
    
    def __getitem__(self, idx):
        lineidx = self.lineidx[idx]
        self.tsv.seek(lineidx)
        item = self.tsv.readline().split('\t')
        
        vid, bufs = item[0], item[1:]
        
        img = []
        for b in bufs:
            img.append(self.str2img(b).unsqueeze(0))
        img = T.cat(img, dim=0)
        
        txt, mask = self.str2txt(self.txt[vid][0])
        
        vq = np.array(sum([[-1]+c.flatten().tolist() for c in self.vq[vid]], []), dtype=np.int64)
        
        return img, txt, mask, vq

def get_dl(ds, size_batch, ep=None):
    sp = T.utils.data.distributed.DistributedSampler(ds, shuffle=(ds.split=='train'))
    if ds.split=='train':
        sp.set_epoch(ep)
    dl = T.utils.data.DataLoader(ds, batch_size=size_batch, num_workers=4, 
                                 pin_memory=True, sampler=sp, worker_init_fn=ds.read_tsv)
    return dl

class VIOLET_Pretrain(VIOLET_Base):
    def __init__(self):
        super().__init__()
        
        self.fc = T.nn.Sequential(*[T.nn.Dropout(0.1), 
                                    T.nn.Linear(768, 768*2), T.nn.ReLU(inplace=True), 
                                    T.nn.Linear(768*2, 1)])
        
        bert = transformers.BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.fc_mtm = bert.cls
        self.fc_mvm = T.nn.Sequential(*[T.nn.Dropout(0.1), 
                                        T.nn.Linear(768, 768*2), T.nn.ReLU(inplace=True), 
                                        T.nn.Linear(768*2, 8192)])
    
    def get_att(self, img, txt, mask):
        feat_img, mask_img, feat_txt, mask_txt = self.go_feat(img, txt, mask)
        _, att = self.go_cross(feat_img, mask_img, feat_txt, mask_txt)
        att = T.cat([a.mean(dim=1, keepdim=True) for a in att], dim=1).sum(dim=(1, 2))
        return att
    
    def forward(self, img, txt, mask):
        (_B, _T, _, _H, _W), (_, _X) = img.shape, txt.shape
        _h, _w = _H//32, _W//32
        _O = min(_B, 4)
        
        feat_img, mask_img, feat_txt, mask_txt = self.go_feat(img, txt, mask)
        out, _ = self.go_cross(feat_img, mask_img, feat_txt, mask_txt)
        out_mtm, out_mvm = self.fc_mtm(out[:, (1+_h*_w)*_T:]), self.fc_mvm(out[:, :(1+_h*_w)*_T])
        
        pdt_feat_img, pdt_mask_img, pdt_feat_txt, pdt_mask_txt = [], [], [], []
        for i in range(_B):
            pdt_feat_img.append(feat_img[i].unsqueeze(0)), pdt_mask_img.append(mask_img[i].unsqueeze(0))
            pdt_feat_txt.append(feat_txt[i].unsqueeze(0)), pdt_mask_txt.append(mask_txt[i].unsqueeze(0))
            
            neg = np.random.permutation([j for j in range(_B) if j!=i])
            for j in range(_O-1):
                j = neg[j]
                pdt_feat_img.append(feat_img[i].unsqueeze(0)), pdt_mask_img.append(mask_img[i].unsqueeze(0))
                pdt_feat_txt.append(feat_txt[j].unsqueeze(0)), pdt_mask_txt.append(mask_txt[j].unsqueeze(0))
        pdt_feat_img, pdt_mask_img, pdt_feat_txt, pdt_mask_txt = [T.cat(x, dim=0) for x in [pdt_feat_img, pdt_mask_img, 
                                                                                            pdt_feat_txt, pdt_mask_txt]]
        out, _ = self.go_cross(pdt_feat_img, pdt_mask_img, pdt_feat_txt, pdt_mask_txt)
        out_vtm = self.fc(out[:, (1+_h*_w)*_T, :]).squeeze().view([_B, _O]) / 0.05
        
        ans_vtm = T.tensor([0 for _ in range(_B)]).long().cuda()
        
        return out_mtm, out_mvm, out_vtm, ans_vtm
    
class Agent_Pretrain(Agent_Base):
    def __init__(self, args, model):
        super().__init__(args, model)
    
    def masking(self, img, txt, mask, vq):
        (_B, _T, _, _H, _W), (_, _X) = img.shape, txt.shape
        _h, _w = _H//32, _W//32
        
        spc_txt, spc_vq = T.logical_or(T.logical_or(txt==101, txt==102), txt==0), (vq==-1)
        spc_all = T.cat([spc_vq, spc_txt], dim=1)
        
        with T.cuda.amp.autocast():
            att = self.model.module.get_att(img.cuda(), txt.cuda(), mask.cuda())
        
        ans_mtm, ans_mvm = T.ones(txt.shape).long()*-1, T.ones(vq.shape).long()*-1
        for i in range(_B):
            if np.random.choice([0, 1])==0: # blockwise masking
                mask_mtm = T.where(T.logical_and(T.logical_not(spc_txt[i]), T.rand(_X)<0.15))[0]
                
                mask_mvm = set()
                for _ in range(_T):
                    t, h, w = [np.random.randint(1, _T) if _T>1 else 1, 
                               np.random.randint(1, _h*2//3), np.random.randint(1, _w*2//3)]
                    t1, h1, w1 = [np.random.randint(0, _T-t+1), 
                                  np.random.randint(0, _h-h+1), np.random.randint(0, _w-w+1)]
                    for i_t in range(t1, t1+t):
                        for i_h in range(h1, h1+h):
                            for i_w in range(w1, w1+w):
                                mask_mvm.add((i_t, i_h, i_w))
                mask_mvm = list(mask_mvm)
                
            else: # attended masking
                att[i][T.where(spc_all[i])] = 0.0
                pos = T.multinomial(att[i], int(((1+_h*_w)*_T+_X)*0.15)).data.cpu().numpy()
                
                mask_mtm, mask_mvm = [], []
                for p in pos:
                    if p<(1+_h*_w)*_T: # img
                        i_t, p = p//(1+_h*_w), p%(1+_h*_w)-1
                        i_h, i_w = p//_w, p%_w
                        mask_mvm.append((i_t, i_h, i_w))
                    else: # txt
                        p -= (1+_h*_w)*_T
                        mask_mtm.append(p)
            
            for p in mask_mtm:
                ans_mtm[i][p], txt[i][p] = txt[i][p], 103
                
            cov = T.zeros(_T, _h, _w)
            for i_t, i_h, i_w in mask_mvm:
                cov[i_t][i_h][i_w] = 1.0
                p = (1+_h*_w)*i_t + 1 + i_h*_w+i_w
                ans_mvm[i][p] = vq[i][p]
            cov = cov.unsqueeze(1).unsqueeze(3).unsqueeze(5).expand([-1, 3, -1, 32, -1, 32])
            cov = cov.flatten(2, 3).flatten(3, 4)
            img[i] *= (1.0-cov)
        
        return img, txt, mask, ans_mtm, ans_mvm
    
    def step(self, img, txt, mask, ans_mtm, ans_mvm, is_train):
        img, txt, mask, ans_mtm, ans_mvm = [x.cuda() for x in [img, txt, mask, ans_mtm, ans_mvm]]
        
        self.optzr.zero_grad()
        with T.cuda.amp.autocast():
            out_mtm, out_mvm, out_vtm, ans_vtm = self.model(img, txt, mask)
            ls_mtm, ls_mvm, ls_vtm = [self.loss_func(o.flatten(0, len(o.shape)-2), a.flatten(0, len(a.shape)-1)) \
                                      for o, a in zip([out_mtm, out_mvm, out_vtm], 
                                                      [ans_mtm, ans_mvm, ans_vtm])]
            ls = ls_mtm+ls_mvm+ls_vtm
        if is_train==True:
            self.scaler.scale(ls).backward()
            self.scaler.step(self.optzr)
            self.scaler.update()
            return {'mtm': ls_mtm.item(), 'mvm': ls_mvm.item(), 'vtm': ls_vtm.item()}
        else:
            out_mtm, out_mvm, out_vtm = [T.argmax(o, dim=-1) for o in [out_mtm, out_mvm, out_vtm]]
            ac_mtm, ac_mvm, ac_vtm = [float((o==a).sum() / (a!=-1).sum()) \
                                      for o, a in zip([out_mtm, out_mvm, out_vtm], 
                                                      [ans_mtm, ans_mvm, ans_vtm])]
            return {'mtm': ac_mtm, 'mvm': ac_mvm, 'vtm': ac_vtm}
    
    def reduce_mean(self, v):
        v = T.tensor(v).cuda()
        DIST.all_reduce(v)
        v = v.item()/DIST.get_world_size()   
        return v
    
    def go_dl(self, dl, is_train):
        ret = {'mtm': [], 'mvm': [], 'vtm': []}
        for img, txt, mask, vq in iter_tqdm(dl):
            img, txt, mask, ans_mtm, ans_mvm = self.masking(img, txt, mask, vq)
            
            try:
                r = self.step(img, txt, mask, ans_mtm, ans_mvm, is_train)
                ret = {k: l+[r[k]] for k, l in ret.items()}
            except:
                print('===== Error step_pretrain on Rank %d ====='%(DIST.get_rank()))
        ret = {k: self.reduce_mean(float(np.average([v for v in l if math.isnan(v)==False]))) for k, l in ret.items()}
        
        return ret
    
if __name__=='__main__':
    dist = Dist()
    
    args = json.load(open('_data/args_pretrain.json', 'r'))
    if DIST.get_rank()==0:
        args['path_output'] = '_snapshot/_pretrain_%s'%(datetime.now().strftime('%Y%m%d%H%M%S'))
        os.makedirs(args['path_output'], exist_ok=True)
        json.dump(args, open('%s/args.json'%(args['path_output']), 'w'), indent=2)
        print(args)
    
    DATASET = ['webvid2.5m', 'cc3m']
    
    log = {dataset: {'ls_vtm': [], 'ls_mtm': [], 'ls_mvm': [], 
                     'ac_vtm': [], 'ac_mtm': [], 'ac_mvm': []} for dataset in DATASET}
    if DIST.get_rank()==0:
        json.dump(log, open('%s/log.json'%(args['path_output']), 'w'), indent=2)
    
    model = VIOLET_Pretrain().cuda()
    model.load_ckpt(args['path_ckpt'])
    model = T.nn.parallel.DistributedDataParallel(model, 
                                                  device_ids=[dist.rank_local], output_device=dist.rank_local, 
                                                  find_unused_parameters=True)
    if DIST.get_rank()==0:
        T.save(model.module.state_dict(), '%s/ckpt_violet_pretrain_0.pt'%(args['path_output']))
    
    agent = Agent_Pretrain(args, model)
    for e in iter_tqdm(range(args['size_epoch'])):
        for dataset in DATASET:
            dl_vl = get_dl(Dataset_Pretrain(args, dataset, 'val'), args['size_batch'])
            
            for part in iter_tqdm(range(2)):
                dl_tr = get_dl(Dataset_Pretrain(args, dataset, 'train', part), args['size_batch'], e+1)
                
                model.train()
                ls_tr = agent.go_dl(dl_tr, True)
                
                model.eval()
                ac_vl = agent.go_dl(dl_vl, False)
                
                if DIST.get_rank()==0:
                    for k in ls_tr:
                        log[dataset]['ls_%s'%(k)].append(ls_tr[k]), log[dataset]['ac_%s'%(k)].append(ac_vl[k])
                    json.dump(log, open('%s/log.json'%(args['path_output']), 'w'), indent=2)
                    T.save(model.module.state_dict(), '%s/ckpt_violet_pretrain_%s_%d_%d.pt'%(args['path_output'], dataset, part, e+1))
                    print('Ep %d:'%(e+1), ls_tr, ac_vl)
                DIST.barrier()
                
