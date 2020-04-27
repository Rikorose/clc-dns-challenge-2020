import argparse
import glob
import math
import os
import time
from multiprocessing.pool import Pool

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F


def resample(x, source_sr, target_sr):
    import warnings

    warnings.filterwarnings("ignore")
    import librosa

    return librosa.resample(x, source_sr, target_sr, scale=True)


def load_audio(file: str, sr: int, verbose=False):
    audio: np.ndarray
    audio, source_sr = sf.read(file)
    if audio.ndim > 1 and audio.shape[-1] > 1:
        raise ValueError("Unsupported number of channels.")
    if source_sr != sr:
        if verbose:
            print(f"Wrong sampling rate. Resampling audio from {source_sr} to {sr}.")
        audio = resample(audio, source_sr, sr)
    return torch.from_numpy(audio).requires_grad_(False)


def enhance_jit(model, audio, verbose=False, sr=16_000):
    timings = []
    model.eval()
    hop = model.hop.item()
    frame_size = model.n_fft.item()

    # Setup buffers
    buf_norm, buf_rnn, buf_clc, buf_ola_wnorm = model.init_buffers()
    # Pad input to the left since we need `clc_order` frames to produce the first output
    # We use reflection pad since this allows a better initialization of the norm, and
    # rnn states.
    clc_order = buf_clc.shape[0]
    audio = F.pad(
        audio.reshape(1, 1, -1), (model.hop.item() * clc_order, 0), mode="reflect"
    ).squeeze()

    start = hop * clc_order
    end = audio.numel()
    n_frames = math.ceil((end - start) / hop)
    # Additionally pad some zeros to the right for the final overlap add iterations
    pad_end = n_frames * hop - audio.numel() + frame_size - hop
    audio = F.pad(audio, (0, pad_end))
    # Setup output buffer
    enhanced = torch.zeros_like(audio)
    # Iterate clc_order to init the buffers before producing an output.
    # This only uses the padded audio and is only necessary for processing finite
    # samples that should have the same output length as the input.
    with torch.no_grad():
        for frame_idx in range(0, start * hop, hop):
            frame = audio[frame_idx : frame_idx + frame_size]
            buf_clc = model.fft_step(frame, buf_clc)
            model_out, buf_norm, buf_rnn = model(buf_clc, buf_norm, buf_rnn)
            frame_enh = torch.zeros_like(frame)
            buf_ola_wnorm = model.ifft_step(model_out, frame_enh, buf_ola_wnorm)
        # Start with an offset of 5 frames
        for frame_idx in range(start, n_frames * hop, hop):
            t0 = time.time()
            frame = audio[frame_idx : frame_idx + frame_size]
            buf_clc = model.fft_step(frame, buf_clc)
            model_out, buf_norm, buf_rnn = model(buf_clc, buf_norm, buf_rnn)
            buf_ola_wnorm = model.ifft_step(
                model_out, enhanced[frame_idx : frame_idx + frame_size], buf_ola_wnorm
            )
            t1 = time.time()
            timings.append(t1 - t0)
    # Remove zero padding at the end
    enhanced[:pad_end]
    # Calculate processing time per frame
    if verbose:
        frame_len_ms = frame_size / sr * 1000
        m_frame_ms = np.mean(timings) * 1000
        print(
            f"Enhanced one frame of length {frame_len_ms} ms in average {m_frame_ms:.2f} ms"
        )
    return enhanced


def worker_init(
    func,
    model_fn,
    verbose=False,
    overwrite=False,
    map_location="cpu",
    num_omp_threads=None,
):
    func.model = torch.jit.load(model_fn, map_location=map_location)
    func.sr = func.model.sr.item()
    func.verbose = verbose
    func.overwrite = overwrite
    if num_omp_threads is not None:
        torch.set_num_threads(num_omp_threads)
        os.environ["OMP_NUM_THREADS"] = str(int(num_omp_threads))


def worker_fn(file):
    fn = os.path.basename(file)
    enh_file = os.path.join(args.output_folder, fn)
    if os.path.isfile(enh_file) and not worker_fn.overwrite:
        print(f"Enhanced audio files {fn} already exits. Skipping.")
        return
    if worker_fn.verbose:
        print(f"Reading file {fn}")
    audio = load_audio(file, worker_fn.sr, args.verbose)
    enhanced = enhance_jit(worker_fn.model, audio, args.verbose, worker_fn.sr)
    if worker_fn.verbose:
        print(f"Writing enhanced audio file to {enh_file}")
    sf.write(enh_file, enhanced.numpy(), worker_fn.sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing enhanced audio files.",
    )
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_omp_threads", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=1)
    parser.add_argument("jit_file", type=str, help="Path to the jit model file.")
    parser.add_argument(
        "input_folder", type=str, help="Folder containing noisy input audio files"
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Output folder. All file names will be the same as in the input folder.",
    )
    args = parser.parse_args()
    if not os.path.isfile(args.jit_file):
        raise FileNotFoundError("Jit file not found")

    if not os.path.exists(args.input_folder):
        raise ValueError("Input folder not found.")

    if args.verbose:
        print(f"Creating output folder {args.output_folder}")
    os.makedirs(args.output_folder, exist_ok=True)

    if args.verbose:
        print(f"Processing audio files in folder {args.input_folder}")
    files = glob.glob(args.input_folder + "*.wav")
    map_location = "cpu"
    with Pool(
        args.num_workers,
        initializer=worker_init,
        initargs=(
            worker_fn,
            args.jit_file,
            args.verbose,
            args.overwrite,
            map_location,
            args.num_omp_threads,
        ),
    ) as p:
        p.map(worker_fn, files, chunksize=args.chunk_size)
