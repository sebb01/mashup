# TODO: accept 2-track stems (acapella, instrumental)

import soundfile as sf
import pyrubberband as pyrb
from enum import Enum
import numpy as np
import os
import simpleaudio as sa
import json
import pathlib

NS_IN_ONE_SECOND = 1000000000
TRACK_PATHS = [file for file in pathlib.Path("songs").rglob("*") if os.path.isfile(file)]
SONG_PATHS = [os.path.join("songs", p) for p in os.listdir("songs") if not os.path.isfile(p)]

class Key(Enum):
    A = 0
    Bb = 1
    B = 2
    C = 3
    Db = 4
    D = 5
    Eb = 6
    E = 7
    F = 8
    Gb = 9
    G = 10
    Ab = 11
    
class Mode(Enum):
    Major = 0
    Minor = 1
    
class Track:
    def __init__(self, song_name, bpm, key, mode, pitchless=["Drums"], wav=None, sr=44100, path=None):
        self.song_name = song_name
        self.bpm = bpm
        self.key = key
        self.mode = mode
        self.pitchless = pitchless
        self.wav = wav
        self.sr = sr
        if self.string is None:
            Exception("Tried to initialize a general Track object; initialize a Track subclass instead")
            
        if wav is None:
            if path == None:
                Exception("Track object needs to be passed either path to a directory or raw audio array")
            self.wav, self.sr = sf.read(str(path))
        try:
            ls = list(pitchless)
        except TypeError:
            pass # pitchless is assumed to be a bool if not iterable
        else:
            pitchless = self.string in ls # pitchless is true iff the trackType is included in the pitchless list
            
    def __repr__(self):
        return f"{self.string} of {self.song_name}"
    
    def __str__(self):
        return self.__repr__()

class Drums(Track):
    def __init__(self, song_name, bpm, key=None, mode=None, pitchless=True, wav=None, sr=44100, path=None):
        self.string = "Drums"
        super().__init__(song_name, bpm, key, mode, pitchless=pitchless, wav=wav, sr=sr, path=path)
        
class Bass(Track):
    def __init__(self, song_name, bpm, key, mode, pitchless=False, wav=None, sr=44100, path=None):
        self.string = "Bass"
        super().__init__(song_name, bpm, key, mode, pitchless=pitchless, wav=wav, sr=sr, path=path)
        
class Other(Track):
    def __init__(self, song_name, bpm, key, mode, pitchless=False, wav=None, sr=44100, path=None):
        self.string = "Other"
        super().__init__(song_name, bpm, key, mode, pitchless=pitchless, wav=wav, sr=sr, path=path)  
        
class Vocals(Track):
    def __init__(self, song_name, bpm, key, mode, pitchless=False, wav=None, sr=44100, path=None):
        self.string = "Vocals"
        super().__init__(song_name, bpm, key, mode, pitchless=pitchless, wav=wav, sr=sr, path=path)
        
class Mashup:
    def __init__(self, wav, name, sr=44100):
        self.wav = wav
        self.name = name
        self.sr = sr
        
    def __repr__(self):
        return self.name
    
    def __str__(self):
        return self.__repr__()
        
    
def add_wav_list(wavs):
    accu = wavs[0]
    if len(wavs) <= 1:
        return accu
    for wav in wavs[1:]:
        accu = add_wavs(accu, wav)
    return accu
    
def stretch_track(track, new_bpm):
    old_bpm = track.bpm
    ratio = new_bpm/old_bpm
    wav_stretch = pyrb.time_stretch(track.wav, track.sr, ratio)
    trackType = globals()[track.string]
    return trackType(track.song_name, new_bpm, track.key, track.mode, wav=wav_stretch, sr=track.sr)

def find_middle_bpm(tracks):
    tempi = [track.bpm for track in tracks]
    return np.mean(tempi)

def find_middle_key(tracks):
    keys = [track.key.value for track in tracks]
    return Key(int(np.mean(keys)))

def add_wavs(wav1, wav2):
    shorter = wav1
    longer = wav2
    if len(wav1) > len(wav2):
        shorter = wav2
        longer = wav1
    diff = len(longer) - len(shorter)
    if diff != 0:
        shorter = np.pad(shorter, ((0, diff), (0, 0)))
    return shorter + longer
    
def transpose_track(track, new_key):
    semitones = new_key.value - track.key.value % 12
    print(f"Transposing {track} by {semitones} semitones to {new_key}...")
    new_wav = pyrb.pitch_shift(track.wav, track.sr, semitones)
    trackType = globals()[track.string]
    return trackType(track.song_name, track.bpm, new_key, track.mode, wav=new_wav, sr=track.sr)

def play_wav_array(array, sr):
    # normalize to 16bit
    array *= 32767 / np.max(np.abs(array))
    # cast to 16bit
    array = array.astype(np.int16)
    waveObject = sa.WaveObject(array, 2, 2, sr)
    # returns sa.PlayObject
    return waveObject.play()      

# song_length: Fraction of 32 bars for the desired song length
# returns Mashup object
def mashup_tracks(tracks, new_bpm=None, new_key=None, song_length=0.5):
    if new_bpm is None:
        new_bpm = find_middle_bpm(tracks)
    if new_key is None:
        new_key = find_middle_key(tracks)
    new_tracks = []
    for track in tracks:
        track.wav = track.wav[:int(song_length*len(track.wav))]
        transposed = transpose_track(track, new_key)
        transposed_stretched = stretch_track(transposed, new_bpm)
        new_tracks.append(transposed_stretched)
    mashup_wav = add_wav_list([track.wav for track in new_tracks])
    mashup_name = make_mashup_name(new_tracks)
    return Mashup(mashup_wav, mashup_name)

def instr_acapella_mashup(instr, acap):
    # TODO
    pass

'''
# Make a random mashup
# track_types: track types ("instruments") to include
# nr_diff_tracks: number of unique songs to draw stems from
# returns Mashup object
def semi_random_mashup(track_types = ["Drums", "Bass", "Other", "Vocals"], nr_diff_tracks=2, bpm=None, key=None):
    if nr_diff_tracks >= len(track_types):
        return random_mashup(track_types, bpm, key)
    
    np.random.shuffle(track_types)
    
    song_paths = SONG_PATHS.copy()
    np.random.shuffle(song_paths)
    for song_path in song_paths:
        if nr_diff_tracks <= 0:
            break
        track_path = 
'''
    
    
def random_mashup(track_types = ["Drums", "Bass", "Other", "Vocals"], bpm=None, key=None, song_length = 0.5):
    tracks = []
    for track_type in track_types:
        tracks.append(load_random_track(track_type))
    return mashup_tracks(tracks, bpm, key, song_length)

def make_mashup_name(tracks):
    names = [track.song_name for track in tracks]
    names = list(dict.fromkeys(names)) # Remove duplicates, from https://www.w3schools.com/python/python_howto_remove_duplicates.asp
    s = ""
    for name in names:
            s += f"{name} x "
    return s[:-3]

def track_length(wav, sr):
    samples_per_ns = sr / NS_IN_ONE_SECOND
    return len(wav) / samples_per_ns

def get_track_string(file_path):
    return os.path.split(file_path)[1][:-4].capitalize()
    
def load_random_track(track_type = None):
    paths = TRACK_PATHS.copy()
    np.random.shuffle(paths)
    if track_type is None:
        return load_track(paths[0])
    for path in paths:
        if get_track_string(path) == track_type:
            return load_track(path)
    raise Exception(f'No tracks of type \"{track_type}\" were found')

def load_track(file):
    directory = os.path.split(file)[0]
    with open(os.path.join(directory, "info.json")) as info_file:
        s = json.load(info_file)
    trackstring = get_track_string(file)
    trackType = globals()[trackstring]
    try:
        pitchless = s["pitchless"]
    except KeyError:
        pitchless = ["Drums"] # If the pitchless key is missing, the default case is assumed
    return trackType(s["name"], s["bpm"], Key(s["key"]), s["mode"], pitchless=pitchless, path=file)
    

mashup = random_mashup()
while True:
    playObject = play_wav_array(mashup.wav, mashup.sr)
    print(f"Now playing: {mashup}")
    mashup = random_mashup()
    # TODO eliminate the delay between tracks
    playObject.wait_done()