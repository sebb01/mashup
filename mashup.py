import soundfile as sf
import pyrubberband as pyrb
from enum import Enum
import numpy as np
from numpy import ndarray
import os
import simpleaudio as sa
import json
import pathlib
from collections import defaultdict
from collections.abc import Iterable
import warnings

# TODO: fix playback pausing while loading next mashup
# TODO: maybe make the song name attribute in info.json not forced unique by using a hash of the wave as ID instead? not sure

NS_IN_ONE_SECOND = 1000000000
SONGDIR = "songs"
STEM_PATHS = [file for file in pathlib.Path(SONGDIR).rglob("*") if os.path.isfile(file)]
SONG_PATHS = [os.path.join(SONGDIR, p) for p in os.listdir(SONGDIR) if not os.path.isfile(p)]
KEYS = {'A':0, 'Bb':1, 'B':2, 'C':3, 'Db':4, 'D':5, 'Eb':6, 'E':7, 'F':8, 'Gb':9, 'G':10, 'Ab': 11}
MODES = ('Minor', 'Major')
    
class Stem:
    """Class that holds the audio data and attributes of one stem of a song"""
    def __init__(self, song_name, bpm, key, mode, pitchless=["Drums"], wav=None, sr=44100, path=None):
        self.song_name = song_name
        self.bpm = bpm
        self.key = key
        self.mode = mode.capitalize()
        self.pitchless = pitchless
        self.wav = wav
        self.sr = sr
        try:
            self.string
        except AttributeError:
            self.string = "Stem"
            
        if wav is None:
            if path == None:
                Exception("Stem object needs to be passed either path to a directory or raw audio array")
            self.wav, self.sr = sf.read(str(path))
        try:
            ls = [string.capitalize() for string in pitchless]
        except TypeError:
            pass # pitchless is assumed to be a bool if not iterable
        else:
            self.pitchless = self.string in ls # pitchless is true iff the stem type is included in the pitchless list

        if not self.pitchless and self.mode not in MODES:
            warnings.warn(f'\"{self.mode}\" is neither Major nor Minor. Stem \"{self.string}\" of \"{self.song_name}\" '+
                f'will only be mashed up with other Stems that use the \"{self.mode}\" mode.')
        

    def __repr__(self):
        return f"{self.string} of {self.song_name}"
    
    def __str__(self):
        return self.__repr__()

    def __eq__(self, o):
        return self.__str__() == o.__str__()

    def __ne__(self, o):
        return not self.__eq__(o)

class Drums(Stem):
    def __init__(self, song_name, bpm, key=None, mode=None, pitchless=True, wav=None, sr=44100, path=None):
        self.string = "Drums"
        super().__init__(song_name, bpm, key, mode, pitchless=pitchless, wav=wav, sr=sr, path=path)
        
class Bass(Stem):
    def __init__(self, song_name, bpm, key, mode, pitchless=False, wav=None, sr=44100, path=None):
        self.string = "Bass"
        super().__init__(song_name, bpm, key, mode, pitchless=pitchless, wav=wav, sr=sr, path=path)
        
class Other(Stem):
    def __init__(self, song_name, bpm, key, mode, pitchless=False, wav=None, sr=44100, path=None):
        self.string = "Other"
        super().__init__(song_name, bpm, key, mode, pitchless=pitchless, wav=wav, sr=sr, path=path)  
        
class Vocals(Stem):
    def __init__(self, song_name, bpm, key, mode, pitchless=False, wav=None, sr=44100, path=None):
        self.string = "Vocals"
        super().__init__(song_name, bpm, key, mode, pitchless=pitchless, wav=wav, sr=sr, path=path)
        
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
        s = f"{self.name}\n"
        for stem in self.stems:
            s = s + f"{stem.string}: {stem.song_name}\n"
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
    """Stretch a stem to a new BPM"""
    old_bpm = stem.bpm
    ratio = new_bpm/old_bpm
    wav_stretch = pyrb.time_stretch(stem.wav, stem.sr, ratio)
    stemType = get_stem_type(stem)
    return stemType(stem.song_name, new_bpm, stem.key, stem.mode, wav=wav_stretch, sr=stem.sr)

def get_stem_type(stem: type[Stem]) -> type:
    """Get the type of stem of a stem"""
    try:
        return globals()[stem.string]
    except:
        return Stem

def find_middle_bpm(stems: Iterable[type[Stem]]) -> float:
    """Find the mean BPM in a list of stems"""
    # TODO: maybe median is better
    tempi = [stem.bpm for stem in stems]
    return np.mean(tempi)

def find_middle_key(stems: Iterable[type[Stem]]) -> int:
    """Find the key that has minimal distance to all stems in a list"""
    # TODO: mean makes no sense since it wraps around after 11, i should use something else
    keys = []
    for stem in stems:
        if not stem.pitchless:
            keys.append(stem.key)
    if len(keys) <= 0:
        return 0
    return int(np.mean(keys))

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
    return stemType(stem.song_name, stem.bpm, new_key, stem.mode, wav=new_wav, sr=stem.sr)

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
        new_stems.append(Stem("N/A (Merged stem)", group[0].bpm, group[0].key, group[0].mode, wav=new_wav))
    return new_stems


def mashup_stems(stems: Iterable[type[Stem]], bpm=None, key:int=None, song_length:float=0.5) -> Mashup:
    """`song_length`: Desired mashup length as a fraction of 32 bars"""
    if bpm is None:
        bpm = find_middle_bpm(stems)
    if key is None:
        key = find_middle_key(stems)

    original_stems = stems.copy()
    stems = merge_same_bpm_and_key(stems)
    nr_samples = int(min([len(stem.wav) for stem in stems]) * song_length)
    transposed = []
    for stem in stems:
        stem.wav = stem.wav[:nr_samples]
        transposed.append(transpose_stem(stem, key))
    transposed = merge_same_bpm_and_key(transposed)

    transposed_stretched = []
    for stem in transposed:
        transposed_stretched.append(stretch_stem(stem, bpm))

    mashup_wav = sum_wav_list([stem.wav for stem in transposed_stretched])
    return Mashup(mashup_wav, original_stems)

def instr_acapella_mashup(instr:str, acap:str, bpm=None, key:int=None, song_length:float=1):
    """`song_length`: Desired mashup length as a fraction of 32 bars"""
    drums = load_stem(os.path.join(SONGDIR, instr, "drums.wav"))
    other = load_stem(os.path.join(SONGDIR, instr, "other.wav"))
    bass = load_stem(os.path.join(SONGDIR, instr, "bass.wav"))
    vocals = load_stem(os.path.join(SONGDIR, acap, "vocals.wav"))

    return mashup_stems([drums, other, bass, vocals], bpm, key, song_length)
    
def random_mashup(stem_types = ["Drums", "Bass", "Other", "Vocals"], bpm=None, key=None, song_length=0.5, allow_duplicates=False):
    stems = []
    for stem_type in stem_types:
        stem = load_random_stem(stem_type)
        if not allow_duplicates:
            while (stem in stems):
                stem = load_random_stem(stem_type)
        stems.append(stem)
    return mashup_stems(stems, bpm, key, song_length)

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

def load_stem(file):
    directory = os.path.split(file)[0]
    with open(os.path.join(directory, "info.json")) as info_file:
        s = json.load(info_file)
    stemstring = get_stem_string(file)
    stemType = globals()[stemstring]
    key, mode, pitchless = get_optional_values(s)
    return stemType(s["name"], s["bpm"], KEYS[key], mode, pitchless=pitchless, path=file)
    

def infinite_random_mashup(stem_types = ["Drums", "Bass", "Other", "Vocals"], bpm=None, key=None, song_length=0.5, allow_duplicates=False):
    mashup = random_mashup(stem_types, bpm, key, song_length, allow_duplicates)
    while True:
        playObject = play_wav_array(mashup.wav, mashup.sr)
        print(f"Now playing: {mashup.description()}")
        mashup = random_mashup(stem_types, bpm, key, song_length, allow_duplicates)
        # TODO eliminate the delay between stems
        # TODO skip button
        playObject.wait_done()

def play_mashup(mashup, print_info = True):
    if print_info:
        print(f"Now playing: {mashup.description()}")
    playObject = play_wav_array(mashup.wav, mashup.sr)
    playObject.wait_done()

def main():
    #import simpleaudio.functionchecks as fc

    #fc.LeftRightCheck.run()
    #print("DONE")
    infinite_random_mashup(song_length=1/8, bpm=105)

    # '''
    # acap = "interior crocodile"
    # instr = "psychosocial verse"
    # mashup = instr_acapella_mashup(instr, acap)
    # play_mashup(mashup)
    # '''

    # drums = load_stem(os.path.join(SONGDIR, "the less", "drums.wav"))
    # other = load_stem(os.path.join(SONGDIR, "beat it", "other.wav"))
    # bass = load_stem(os.path.join(SONGDIR, "psychosocial chorus", "bass.wav"))
    # vocals = load_stem(os.path.join(SONGDIR, "psychosocial chorus", "vocals.wav"))
    # mashup = mashup_stems([drums, other, bass, vocals])
    # play_mashup(mashup)

if __name__ == "__main__":
    main()