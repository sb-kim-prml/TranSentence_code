import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
import random

import os
import argparse
import glob
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

import torchaudio.transforms as T

import soundfile as sf
import torchaudio
import librosa

def check_sr(wav):
    sr = sf.info(wav).samplerate
    return sr


def main(args):
    n_gpus = torch.cuda.device_count()
    port = 50000 + random.randint(0,1000)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, args))

    split = ['train', 'valid', 'test']
    for split in splits:
        wavs_dir = os.path.join(args.output_dir, split)
        wavs = glob.glob(os.path.join(wavs_dir, '*.wav'))
        with open(os.path.join(args.output_dir, f"unit_manifest_{split}.txt"), 'w') as f:
            f.write(f"{wavs_dir}\n")
            for wav in wavs:
                f.write(f"{wav}\t0\n")



def run(rank, n_gpus, args):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(1234)
    torch.cuda.set_device(rank)

    resamplers = {}
    for source_sr in [24000, 32000, 44100, 48000]:
        resamplers[source_sr] = T.Resample(source_sr, args.target_sr, resampling_method="kaiser_window").cuda(rank)

    dset = DLoader(args, args.input_dir)
    d_sampler = torch.utils.data.distributed.DistributedSampler(
        dset,
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True)


    collate_fn = Collate()
    d_loader = DataLoader(dset, num_workers=16, shuffle=False,
                          batch_size=1, pin_memory=True,
                          drop_last=False, collate_fn=collate_fn, sampler=d_sampler)


    extract(d_loader, resamplers, rank, args)


class DLoader():
    def __init__(self, args, input_dir):
        self.wavs = sorted(glob.glob(os.path.join(input_dir, '**/*{}'.format(args.ext)), recursive=True), reverse=True)

        self.output_dir = args.output_dir
        self.trim = args.trim
        print('wav num: ', len(self.wavs))

    def __getitem__(self, index):
        audio, sr = torchaudio.load(self.wavs[index])
        split = os.path.basename(os.path.dirname(self.wavs[index]))
        if split == 'dev':
            split = 'valid'
        new = os.path.join(self.output_dir, split, os.path.basename(self.wavs[index]).split('.')[0] + '.wav')
        if os.path.isfile(new):
            return None, None, None
        os.makedirs(os.path.dirname(new), exist_ok=True)
        if self.trim:
            audio, _ = librosa.effects.trim(audio.squeeze(),
                                            top_db=20,
                                            frame_length=1280,
                                            hop_length=320)
            audio = audio.unsqueeze(0)
        return audio, sr, new

    def __len__(self):
        return len(self.wavs)


class Collate():
    def __init__(self):
        pass

    def __call__(self, batch):
        return batch[0][0], batch[0][1], batch[0][2]


def extract(d_loader, resamplers, rank, args):
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(d_loader)):
            audio, sr, new = batch
            if new == None:
                continue
            audio = audio.cuda(rank)

            if sr != args.target_sr:
                audio = resamplers[sr](audio)
            audio = audio / torch.max(audio.abs()) * 0.95 * 32768.0
            audio = audio.to(dtype=torch.int16)
            torchaudio.save(new, audio.cpu(), args.target_sr, encoding="PCM_S", bits_per_sample=16)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', required=True)
    parser.add_argument('-o', '--output_dir', required=True)
    parser.add_argument('-e', '--ext', default='.wav')
    parser.add_argument('--target_sr', default=16000)
    parser.add_argument('--trim', action='store_true')
    a = parser.parse_args()

    main(a)