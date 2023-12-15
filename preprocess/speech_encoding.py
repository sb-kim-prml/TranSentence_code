import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import argparse
import glob
from tqdm import tqdm
import numpy as np
from random import randint as rdint
import soundfile as sf
import torchaudio
import torchaudio.transforms as T
from fairseq.models.wav2vec import Wav2VecLaser
import torch.nn.functional as F
import librosa

from torchaudio.transforms import MelSpectrogram

def main(args):
    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    port = 60000 + rdint(0, 100)
    os.environ['MASTER_PORT'] = str(port)


    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, args, ))

def same(audio):
    return audio

def run(rank, n_gpus, args):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(1234)
    torch.cuda.set_device(rank)


    dset = DLoader(args)
    d_sampler = torch.utils.data.distributed.DistributedSampler(dset,
                                                                num_replicas=n_gpus,
                                                                rank=rank,
                                                                shuffle=True)
    collate_fn = Collate()
    d_loader = DataLoader(dset, num_workers=16, shuffle=False,
                            batch_size=1, pin_memory=True,
                            drop_last=False, collate_fn=collate_fn, sampler=d_sampler)

    c_dir = os.path.dirname(args.ckpt)
    c_name = os.path.basename(args.ckpt)

    model = Wav2VecLaser.from_pretrained(c_dir, checkpoint_file=c_name).models[0].cuda(rank)
    model.eval()

    extract(rank, d_loader, model)


def extract(rank, d_loader, model):
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(d_loader)):
            audio, new = batch
            if not new:
                continue
            else:
                audio = audio.cuda(rank).unsqueeze(0)

                audio = audio / torch.abs(audio).max() * 0.95

                feats = F.layer_norm(audio, audio.shape)
                padding_mask = torch.Tensor([False] * feats.shape[1])
                sample = {'padding_mask': padding_mask, 'source': feats}
                emb = model(**sample)

                emb_name = new

                os.makedirs(os.path.dirname(emb_name), exist_ok=True)


                torch.save(emb.squeeze(0).cpu().detach(), emb_name)

class DLoader():
    def __init__(self, args):
        self.ext_name = args.ext_name
        self.input_dir = args.input_dir
        self.output_dir = args.output_dir

        self.wavs = sorted(glob.glob(os.path.join(self.input_dir, '**/*{}'.format(self.ext_name)), recursive=True))
        self.base = os.path.basename(self.input_dir)
        self.output_dir = self.output_dir
        print('wav num: ', len(self.wavs))






    def __getitem__(self, index):
        name = self.wavs[index]

        new = os.path.join(self.output_dir, os.path.basename(self.wavs[index]).replace(self.ext_name, '.pt'))

        split = os.path.basename(os.path.dirname(self.wavs[index]))
        new = os.path.join(self.output_dir, split, os.path.basename(self.wavs[index]).replace(self.ext_name, '.pt'))

        if os.path.isfile(new):
            return False, False
        audio, sr = torchaudio.load(self.wavs[index])
        assert sr == 16000
        audio = torch.FloatTensor(audio.squeeze())

        return audio, new


    def __len__(self):
        return len(self.wavs)


class Collate():
    def __init__(self):
        pass

    def __call__(self, batch):
        return batch[0][0], batch[0][1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', required=True)
    parser.add_argument('-o', '--output_dir', default='')
    parser.add_argument('-t', '--target_sr', default=16000)
    parser.add_argument('--ext_name', default='.wav')
    parser.add_argument('-c', '--ckpt', default='./ckpts/english.pt', help='./ckpts/romance.pt | ./ckpts/english.pt')
    a = parser.parse_args()

    if a.output_dir == '':
        a.output_dir = os.path.join(os.path.dirname(a.input_dir.rstrip('/')), 'sem')

    main(a)