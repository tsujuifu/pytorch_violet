
from lib import *

class Agent_Base:
    def __init__(self, args, model):
        super().__init__()
        
        self.args, self.model = args, model
        
        self.loss_func = T.nn.CrossEntropyLoss(ignore_index=-1).cuda()
        self.optzr = T.optim.AdamW(self.model.parameters(), lr=args['lr'], 
                                   betas=(0.9, 0.98), weight_decay=args['decay'])
        self.scaler = T.cuda.amp.GradScaler()
        