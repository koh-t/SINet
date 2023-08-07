import os
import yaml
from datetime import datetime
from glob import glob
model = 'GWN-GWN-MLP'
_path = f'./config/{model}/'
savepath = f'{_path}script/'
if not os.path.exists(savepath):
    os.mkdir(savepath)

cfg = yaml.load(open(f'{_path}/template.yml'), Loader=yaml.FullLoader)
cfg['model'] = model

rep = [16, 32]
out = [64, 128]
hsic = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
alpha = [0.0, 1e-1, 1.0]
mono = [1e-6, 1e-7, 1e-8]

c = len(glob(savepath+'*.yml'))
for _rep in rep:
    cfg['rep_hidden'] = [_rep, _rep]
    for _out in out:
        cfg['out_hidden'] = [_out, _out]
        for _hsic in hsic:
            cfg['hsic'] = _hsic
            for _mono in mono:
                cfg['mono'] = _mono
                for _alpha in alpha:
                    cfg['alpha'] = _alpha
                    filename = 'script_{}_{}_{}.yml'.format(
                        cfg['model'],
                        c,
                        datetime.now().strftime('%Y%m%d-%H:%M:%S:%f'))
                    print(filename)

                    with open(f'{savepath}{filename}', 'w') as outfile:
                        yaml.dump(cfg, outfile, default_flow_style=False)
                    c += 1
