# -*- coding: utf-8 -*-
'''
@author: syfly007
@contact: syfly007@163.com
@file: speechRec.py
@time: 2022/4/10 15:35
@desc:
'''

import os
import shutil

import paddle
from paddlespeech.cli.asr.infer import ASRExecutor
from paddlespeech.cli.text.infer import TextExecutor
import wave
from glob import glob
import re
import click
from datetime import datetime

deal_dir = f'./tmp_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
SEG_SECORDS = 40
asr_executor = ASRExecutor()
text_executor = TextExecutor()


def get_wav_time(wav):
    f = wave.open(wav)
    rate = f.getframerate()
    frames = f.getnframes()
    duration = frames / float(rate)  # 单位为s
    return duration


def wav_to_text(wav):
    text = asr_executor(
        model='conformer_wenetspeech',
        lang='zh',
        sample_rate=16000,
        config=None,  # Set `config` and `ckpt_path` to None to use pretrained model.
        ckpt_path=None,
        audio_file=wav,
        force_yes=True,
        device=paddle.get_device())
    # print('ASR Result: \n{}'.format(text))

    if len(text) <= 0:
        return ''
    result = text_executor(
        text=text,
        task='punc',
        model='ernie_linear_p7_wudao',
        lang='zh',
        config=None,
        ckpt_path=None,
        punc_vocab=None,
        device=paddle.get_device())
    print('Text Result: \n{}'.format(result))
    return result


def mp4_to_wav(mp4_path, wav_path, sampling_rate=16000):
    if os.path.exists(wav_path):
        os.remove(wav_path)
    command = f'ffmpeg -i {mp4_path} -ac 1 -ar {sampling_rate} {wav_path} -y'
    print('cmd：', command)
    os.system(command)

def mp3_to_wav(mp3_path, wav_path, sampling_rate=16000):
    if os.path.exists(wav_path):
        os.remove(wav_path)
    command = f'ffmpeg -i {mp3_path} -ac 1 -ar {sampling_rate} {wav_path} -y'
    print('cmd：', command)
    os.system(command)


def split_wav(wav, name, seg_time=60):
    assert os.path.exists(wav)

    command = f'ffmpeg -i {wav} -f segment -segment_time {seg_time} -c copy {name}%04d.wav'
    print('cmd：', command)
    os.system(command)


def process_one_mp4(mp4):
    print(f'deal mp4:{mp4}')

    basename = str(mp4).split('.mp4')[0]
    mp4 = re.escape(mp4)

    text = f'{basename}.txt'
    if os.path.exists(text):
        print(f'text alreay exist:{text}, pass')
        return

    wav = os.path.join(deal_dir, 'tmp.wav')
    mp4_to_wav(mp4, wav)

    if not os.path.exists(wav):
        print('mp4_to_wav fail')
        return

    process_one_wav(wav, text)
    os.remove(wav)

def process_one_mp3(mp3):
    print(f'deal mp3:{mp3}')

    basename = str(mp3).split('.mp3')[0]
    mp3 = re.escape(mp3)

    text = f'{basename}.txt'
    if os.path.exists(text):
        print(f'text alreay exist:{text}, pass')
        return

    wav = os.path.join(deal_dir, 'tmp.wav')
    mp3_to_wav(mp3, wav)

    if not os.path.exists(wav):
        print('mp3_to_wav fail')
        return

    process_one_wav(wav, text)
    os.remove(wav)

def process_one_wav(wav, text=''):
    print(f'deal one wav:{wav}')
    if not os.path.exists(wav):
        print(f'wav not exist:{wav}, pass')
        return

    if not text:
        basename = str(wav).split('.wav')[0]
        text = f'{basename}.txt'
        if os.path.exists(text):
            print(f'text alreay exist:{text}, pass')
            return

    secords = get_wav_time(wav)

    print(f'secords:{secords}')
    if secords > SEG_SECORDS:
        split_name = os.path.join(deal_dir, f'tmp_split_')
        split_wav(wav, split_name, seg_time=SEG_SECORDS)
        wavs = glob(os.path.join(deal_dir, 'tmp_split_*.wav'))
        wavs.sort()
    else:
        wavs = [wav]

    print(f'wavs:{wavs}')

    t = ''
    for w in wavs:
        print(f'deal wav:{w}')
        t += wav_to_text(w)
        with open(text, 'w') as f:
            print(f'current text:{t}')
            f.write(t)
            f.flush()

    clean_wavs = glob(os.path.join(deal_dir, 'tmp_split_*.wav'))
    for c in clean_wavs:
        os.remove(c)

    print(f'save text in {text}')

def process_mp4(root_path):
    assert os.path.exists(root_path)
    if os.path.isfile and str(root_path).endswith('.mp4'):
        process_one_mp4(root_path)
        return

    mp4s = glob(os.path.join(root_path, '**/*.mp4'), recursive=True)
    for mp4 in mp4s:
        process_one_mp4(mp4)

def process_mp3(root_path):
    assert os.path.exists(root_path)
    if os.path.isfile and str(root_path).endswith('.mp3'):
        process_one_mp3(root_path)
        return

    mp3s = glob(os.path.join(root_path, '**/*.mp3'), recursive=True)
    for mp3 in mp3s:
        process_one_mp3(mp3)

def process_wav(root_path):
    assert os.path.exists(root_path)
    if os.path.isfile and str(root_path).endswith('.wav'):
        process_one_wav(root_path)
        return

    wavs = glob(os.path.join(root_path, '**/*.wav'), recursive=True)
    for w in wavs:
        process_one_wav(w)


@click.command()
@click.option('--scan_path', '-s',
              default='./',
              prompt='media path',
              help='media path')
def cmd(scan_path):
    os.mkdir(deal_dir)

    process_mp3(scan_path)
    process_mp4(scan_path)
    process_wav(scan_path)


def test():
    # t = ''
    # t+=wav_to_text('./audio00.wav')
    # t+=wav_to_text('./audio01.wav')
    # t+=wav_to_text('./audio02.wav')
    # print(t)

    # mp4_to_wav('怎样选择成长股.mp4','test.wav')

    # print(get_wav_time('test.wav'))

    # split_wav('test.wav','test_split_',seg_time=120)
    pass

if __name__ == '__main__':

    cmd()
