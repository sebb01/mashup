import soundfile as sf
import pyrubberband as pyrb
import numpy as np
from numpy import ndarray
import os
import simpleaudio as sa
import json
import pathlib
from collections import defaultdict
from collections.abc import Iterable
import warnings
from inputimeout import inputimeout, TimeoutOccurred
from contextlib import redirect_stdout
from multiprocessing.pool import ThreadPool
import re

# TODO: fix the transition between mashups having a pause
# TODO: maybe make the song name attribute in info.json not forced unique by using some has as an ID
# TODO: Mode where the user can type which stems should stay, rest gets a new random stem
# TODO: move some functions into the Stem class
# TODO: key finding algo has a bug, check mashup between constellation and gohouteki or antonymph x the less i know

NS_IN_ONE_SECOND = 1000000000
SONGDIR = "songs"
STEM_PATHS = [file for file in pathlib.Path(SONGDIR).rglob("*.[wW][aA][vV]")]
SONG_PATHS = [os.path.join(SONGDIR, p) for p in os.listdir(SONGDIR) if not os.path.isfile(p)]
KEYS = {'A':0, 'Bb':1, 'B':2, 'C':3, 'Db':4, 'D':5, 'Eb':6, 'E':7, 'F':8, 'Gb':9, 'G':10, 'Ab': 11}
MODES = ('Minor', 'Major')
    
class Stem:
    """Class that holds the audio data and attributes of one stem of a song"""
    def __init__(self, song_name, bpm, key, mode, pitchless=["Drums"], wav=None, sr=44100, path=None, halftime=False, doubletime=False, balance=1):
        self.song_name = song_name
        self.bpm = bpm
        self.key = key
        self.mode = mode.capitalize()
        self.wav = wav
        self.sr = sr
        self.halftime = halftime
        self.doubletime = doubletime
        try:
            self.string
        except AttributeError:
            self.string = "Stem"
            
        if wav is None:
            if path == None:
                Exception("Stem object needs to be passed either path to a directory or raw audio array")
            self.wav, self.sr = sf.read(str(path))
            self.wav = balance * self.wav
        try:
            ls = [s.capitalize() for s in pitchless]
            self.pitchless = self.string in ls
        except:
            self.pitchless = self.string == "Drums"

        if not self.pitchless and self.mode not in MODES:
            warnings.warn(f'\"{self.mode}\" is neither Major nor Minor. Stem \"{self.string}\" of \"{self.song_name}\" '+
                f'will only be mashed up with other Stems that use the \"{self.mode}\" mode.')
        
        # Treat major keys as their relative minor counterparts
        if self.mode == "Major":
            self.mode = "Minor"
            self.key -= 3
        

    def __repr__(self):
        return f"{self.string}:\t\t{self.song_name}"
    
    def __str__(self):
        return self.__repr__()

    def __eq__(self, o):
        return self.__str__() == o.__str__()

    def __ne__(self, o):
        return not self.__eq__(o)

class Drums(Stem):
    def __init__(self, *args, **kwargs):
        self.string = "Drums"
        super().__init__(*args, **kwargs)

class Bass(Stem):
    def __init__(self, *args, **kwargs):
        self.string = "Bass"
        super().__init__(*args, **kwargs)

class Other(Stem):
    def __init__(self, *args, **kwargs):
        self.string = "Other"
        super().__init__(*args, **kwargs)

class Vocals(Stem):
    def __init__(self, *args, **kwargs):
        self.string = "Vocals"
        super().__init__(*args, **kwargs)
    
    def __repr__(self):
        return f"Vocals/Lead:\t{self.song_name}"
        
class Mashup:
    # TODO: Make this a subclass of Stem
    """Class that holds the audio data and attributes of a mashup."""
    def __init__(self, wav, stems, sr=44100):
        self.wav = wav
        self.stems = stems
        self.name = make_mashup_name(stems)
        self.sr = sr
        for stem in self.stems:
            stem.wav = None    # Purge stem wavs to hopefully free memory
        
    def __repr__(self):
        return self.name
    
    def __str__(self):
        return self.__repr__()

    def description(self):
        s = f"{self.name}\n=====================================================\n"
        for stem in self.stems:
            s = s + str(stem) + "\n"
        return s
        
    
def sum_wav_list(wavs: Iterable[ndarray]) -> ndarray:
    """Sum a list of audio arrays into one audio array"""
    accu = wavs[0]
    if len(wavs) <= 1:
        return accu
    for wav in wavs[1:]:
        accu = sum_wavs(accu, wav)
    return accu
    
def stretch_stem(stem: type[Stem], new_bpm) -> type[Stem]:
    """Stretch a stem to a new BPM.\n
    If the stem is marked as doubletime or halftime it will also be cut in half or repeated twice respectively."""
    wav = stem.wav
    old_bpm = stem.bpm
    ratio = new_bpm/old_bpm

    if stem.doubletime:
        wav = wav[:len(wav)//2]
    wav_stretch = pyrb.time_stretch(wav, stem.sr, ratio)
    if stem.halftime:
        wav_stretch = np.concatenate((wav_stretch, wav_stretch))
    
    stemType = get_stem_type(stem)
    return stemType(
        stem.song_name, new_bpm, stem.key, stem.mode, wav=wav_stretch, sr=stem.sr,
        halftime=stem.halftime, doubletime=stem.doubletime)

def get_stem_type(stem: type[Stem]) -> type:
    """Get the type of stem of a stem"""
    try:
        return globals()[stem.string]
    except:
        return Stem

def find_middle_bpm(stems: Iterable[type[Stem]]) -> float:
    """Find the mean BPM in a list of stems"""
    tempi = [stem.bpm for stem in stems]
    if len(np.unique(tempi)) <= 1: return tempi[0]
    best_tempo_list = []
    best_mask = 0
    min_std = float('inf')

    # Make all possible bitstrings of length len(tempi)
    # 0 means we consider the original BPM, 1 means doubletime
    for mask in range(2**len(tempi)):
        tempi_mixed = []
        for i in range(len(tempi)):
            if (mask>>i) & 1:
                tempi_mixed.append(tempi[i]*2)
            else:
                tempi_mixed.append(tempi[i])
        std = np.std(tempi_mixed)
        if std < min_std:
            min_std = std
            best_tempo_list = tempi_mixed
            best_mask = mask
    
    # If the majority of stems are marked as doubletime, it makes more sense to mark the rest as halftime instead
    if bin(best_mask).count("1") > len(stems) / 2:
        best_tempo_list = np.array(best_tempo_list) / 2
        for i, stem in enumerate(stems):
            if not (best_mask>>i) & 1:
                stem.halftime = True
                stem.bpm = stem.bpm / 2
    else:
        for i, stem in enumerate(stems):
            if (best_mask>>i) & 1:
                stem.doubletime = True
                stem.bpm = stem.bpm * 2
    return np.mean(best_tempo_list) # TODO maybe median is better?

def find_middle_key(stems: Iterable[type[Stem]]) -> int:
    """Find the key that minimizes the maximum distance to any of the stems"""
    keys = [stem.key % 12 for stem in stems if not stem.pitchless]
    if len(keys) <= 0:
        return 0
    min_max_dist = 9999
    best_key = 0
    for candidate in range(12):
        max_dist = max(  min( abs(candidate - k), 12 - abs(candidate - k) ) for k in keys  )
        if max_dist < min_max_dist:
            min_max_dist = max_dist
            best_key = candidate
    return best_key

def sum_wavs(wav1: ndarray, wav2: ndarray) -> ndarray:
    """Sum two audio arrays"""
    shorter = wav1
    longer = wav2
    if len(wav1) > len(wav2):
        shorter = wav2
        longer = wav1
    diff = len(longer) - len(shorter)
    if diff != 0:
        shorter = np.pad(shorter, ((0, diff), (0, 0)))
    return shorter + longer
    
def transpose_stem(stem: type[Stem], new_key: int) -> type[Stem]:
    """Transpose a stem to a new key, unless the stem is marked as pitchless"""
    if stem.pitchless:
        return stem
    semitones = new_key - stem.key % 12
    #print(f"Transposing {stem} by {semitones} semitones to {new_key}...")
    new_wav = pyrb.pitch_shift(stem.wav, stem.sr, semitones)
    stemType = get_stem_type(stem)
    dummy = stemType(stem.song_name, stem.bpm, new_key, stem.mode, wav=new_wav, sr=stem.sr,
            halftime=stem.halftime, doubletime=stem.doubletime)
    return stemType(stem.song_name, stem.bpm, new_key, stem.mode, wav=new_wav, sr=stem.sr,
            halftime=stem.halftime, doubletime=stem.doubletime)

def play_wav_array(array: ndarray, sr: int):
    # normalize to 16bit
    array *= 32767 / np.max(np.abs(array))
    # cast to 16bit
    array = array.astype(np.int16)
    waveObject = sa.WaveObject(array, 2, 2, sr)
    # returns sa.PlayObject
    return waveObject.play()      

def merge_same_bpm_and_key(stems: Iterable[type[Stem]]) -> list[Stem]:
    new_stems = []
    groups = defaultdict(list)
    for stem in stems:
        key = stem.key
        if stem.pitchless:
            key = 'pitchless'
        groups[(stem.bpm, key)].append(stem)
    for group in groups.values():
        if len(group) <= 1:
            new_stems.append(group[0])
            continue
        new_wav = sum_wav_list([stem.wav for stem in group])
        halftime = any(stem.halftime for stem in group)
        doubletime = any(stem.doubletime for stem in group)
        new_stems.append(Stem(make_mashup_name(group), group[0].bpm, group[0].key, group[0].mode, 
                              wav=new_wav, halftime=halftime, doubletime=doubletime))
    return new_stems


def mashup_stems(stems: Iterable[type[Stem]], bpm=None, key:int=None, song_length:float=1) -> Mashup:
    """`song_length`: Desired mashup length as a fraction of 32 bars"""
    if bpm is None:
        bpm = find_middle_bpm(stems)
    if key is None:
        key = find_middle_key(stems)

    original_stems = stems.copy()
    stems = merge_same_bpm_and_key(stems)

    for stem in stems:
        stem.wav = stem.wav[:int(song_length*len(stem.wav))]
    with ThreadPool() as pool:
        transposed = pool.starmap(transpose_stem, [(stem, key) for stem in stems])
    transposed = merge_same_bpm_and_key(transposed)

    with ThreadPool() as pool:
        transposed_stretched = pool.starmap(stretch_stem, [(stem, bpm) for stem in transposed])

    mashup_wav = sum_wav_list([stem.wav for stem in transposed_stretched])
    return Mashup(mashup_wav, original_stems)

def vocalswap_mashup(instr:str, acap:str, balances=[1]*4, **kwargs):
    """`song_length`: Desired mashup length as a fraction of 32 bars"""
    stem_paths = [
        os.path.join(instr, "drums.wav"),
        os.path.join(instr, "other.wav"),
        os.path.join(instr, "bass.wav"),
        os.path.join(acap, "vocals.wav"),
    ]
    stems = []
    for path, balance in zip(stem_paths, balances):
        try:
            stems.append(load_stem(path, balance))
        except:
            print(f"{path} does not exist, so it won't be loaded.")
            pass            
    return mashup_stems(stems, **kwargs)
    
def random_mashup(stem_types = ["Drums", "Bass", "Other", "Vocals"], allow_duplicates=False, **kwargs):
    stems = []
    for stem_type in stem_types:
        stem = load_random_stem(stem_type)
        if not allow_duplicates:
            while (stem in stems):
                stem = load_random_stem(stem_type)
        stems.append(stem)
    return mashup_stems(stems, **kwargs)

def get_random_instrumental_path() -> str:
    pattern = re.compile(r"(?i)(?<!vocals)\.wav$")
    for i in range(500):
        song_path = np.random.choice(SONG_PATHS)
        stem_files = [file for file in pathlib.Path(song_path).rglob("*") if pattern.search(str(file))]
        if stem_files: return song_path
    raise Exception(f"Could not find an instrumental stem after looking through {i+1} song folders")

def get_random_vocal_path() -> str:
    for i in range(500):
        song_path = np.random.choice(SONG_PATHS)
        vocals_path = os.path.join(song_path, "vocals.wav")
        if os.path.exists(vocals_path): return song_path
    raise Exception(f"Could not find a vocal stem after looking through {i+1} song folders")

def random_vocalswap_mashup(_=None, **kwargs):
    # dummy argument is a workaround so that this function can be used interchangeably with random_mashup
    instr = get_random_instrumental_path()
    acap = get_random_vocal_path()
    return vocalswap_mashup(instr, acap, **kwargs)

def make_mashup_name(stems):
    names = [stem.song_name for stem in stems]
    names = list(dict.fromkeys(names)) # Remove duplicates, from https://www.w3schools.com/python/python_howto_remove_duplicates.asp
    s = ""
    for name in names:
            s += f"{name} x "
    return s[:-3]

def stem_length(wav, sr):
    samples_per_ns = sr / NS_IN_ONE_SECOND
    return len(wav) / samples_per_ns

def get_stem_string(file_path):
    return os.path.split(file_path)[1][:-4].capitalize()
    
def load_random_stem(stem_type = None):
    paths = STEM_PATHS.copy()
    np.random.shuffle(paths)
    if stem_type is None:
        return load_stem(paths[0])
    for path in paths:
        if get_stem_string(path) == stem_type:
            return load_stem(path)
    raise Exception(f'No stems of type \"{stem_type}\" were found')

def get_optional_values(song_dict):
    try:
        key = song_dict["key"]
    except KeyError:
        key = "A"

    try:
        mode = song_dict["mode"]
    except KeyError:
        mode = "Minor"

    try:
        pitchless = song_dict["pitchless"]
    except KeyError:
        pitchless = ["Drums"]

    return key, mode, pitchless

def load_stem(file, balance=1):
    directory = os.path.split(file)[0]
    with open(os.path.join(directory, "info.json"), encoding='utf-8') as info_file:
        s = json.load(info_file)
    stemstring = get_stem_string(file)
    stemType = globals()[stemstring]
    key, mode, pitchless = get_optional_values(s)
    return stemType(s["name"], s["bpm"], KEYS[key], mode, pitchless=pitchless, path=file, balance=balance)
    

def infinite_random_mashup(stem_types = ["Drums", "Bass", "Other", "Vocals"], vocalswap=True, **kwargs):
    """Play random mashups indefinitely\n
    `vocalswap`: When set to true all instrumental sources will come from the same song"""

    make_mashup = random_mashup
    if vocalswap: make_mashup = random_vocalswap_mashup

    print("Generating first mashup, please wait...\nPressing enter will skip the current mashup\n")
    with ThreadPool() as pool:
        mashup = pool.apply(make_mashup, args=(stem_types,), kwds=kwargs)
    # TODO: prepare a queue of a couple of mashups with the thread pool, for smoother skipping
    while True:
        playObject = play_wav_array(mashup.wav, mashup.sr)
        print_now_playing(mashup)
        with ThreadPool() as pool:
            result = pool.apply_async(make_mashup, args=(stem_types,), kwds=kwargs)
            wait_or_skip(playObject)
            mashup = result.get()

def play_mashup(mashup, print_info = True):
    if print_info: print_now_playing(mashup)
    playObject = play_wav_array(mashup.wav, mashup.sr)
    playObject.wait_done()

def wait_or_skip(playObject: sa.PlayObject):
    """Wait for a playObject to finish playing, skip the wait if there is any user input"""
    user_input = None
    while playObject.is_playing() and user_input is None:
        try:
            # Redirect stdout because inputimeout() has print() calls that I don't want
            with open(os.devnull, 'w') as f:
                with redirect_stdout(f):
                    user_input = inputimeout(timeout=0.1)
        except TimeoutOccurred:
            pass
    if user_input is not None:
        print("Mashup Skipped...\n")
    playObject.stop()

def find_song(name: str) -> str:
    # TODO support search through song name metadata
    matches = list(pathlib.Path(SONGDIR).rglob(f"*{name}*"))
    if matches:
        return str(matches[0])
    else:
        raise FileNotFoundError(f"No song found matching '{name}'")

def find_stem(song_name: str, stem_name: str):
    return os.path.join(find_song(song_name), stem_name + ".wav")

def print_now_playing(mashup: Mashup):
    print(f"Now playing:\t{mashup.description()}")

def quick_mashup(drums, other, bass, vocals, **kwargs):
    print(loading_message())
    drums = load_stem(find_stem(drums, "drums"))
    other = load_stem(find_stem(other, "other"))
    bass = load_stem(find_stem(bass, "bass"))
    vocals = load_stem(find_stem(vocals, "vocals"))
    play_mashup(mashup_stems([drums, other, bass, vocals], **kwargs))

def quick_vocalswap_mashup(instr, acap, **kwargs):
    print(loading_message())
    play_mashup(vocalswap_mashup(find_song(instr), find_song(acap), **kwargs))

def loading_message():
    return "Preparing Mashup..."

def main():
    #quick_vocalswap_mashup('money', 'const')
    
    infinite_random_mashup()

    # play_mashup(mashup_stems([load_stem(find_stem('money', 'drums')),
    #                           load_stem(find_stem('ocean', 'drums')),
    #                           load_stem(find_stem('little', 'bass')),
    #                           load_stem(find_stem('sword', 'other')),
    #                           load_stem(find_stem('little', 'other'))]))

if __name__ == "__main__":
    main()