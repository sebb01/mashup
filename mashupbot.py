# TODO: accept 2-track stems (acapella, instrumental)

import soundfile as sf
import pyrubberband as pyrb
from enum import Enum
import numpy as np
import os
import simpleaudio as sa
import json
import pathlib
from collections import defaultdict
import warnings

NS_IN_ONE_SECOND = 1000000000
TRACK_PATHS = [file for file in pathlib.Path("songs").rglob("*") if os.path.isfile(file)]
SONG_PATHS = [os.path.join("songs", p) for p in os.listdir("songs") if not os.path.isfile(p)]

KEYS = {'A':0, 'Bb':1, 'B':2, 'C':3, 'Db':4, 'D':5, 'Eb':6, 'E':7, 'F':8, 'Gb':9, 'G':10, 'Ab': 11}
MODES = ('Minor', 'Major')
    
class Track:
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
            self.string = "Track"
            
        if wav is None:
            if path == None:
                Exception("Track object needs to be passed either path to a directory or raw audio array")
            self.wav, self.sr = sf.read(str(path))
        try:
            ls = list(pitchless)
        except TypeError:
            pass # pitchless is assumed to be a bool if not iterable
        else:
            self.pitchless = self.string in ls # pitchless is true iff the trackType is included in the pitchless list

        if not self.pitchless and self.mode not in MODES:
            warnings.warn(f'\"{self.mode}\" is neither Major nor Minor. Track \"{self.string}\" of \"{self.song_name}\" '+
                f'will only be mashed up with other tracks that use the \"{self.mode}\" mode.')
        

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
    def __init__(self, wav, tracks, sr=44100):
        self.wav = wav
        self.tracks = tracks
        self.name = make_mashup_name(tracks)
        self.sr = sr
        for track in self.tracks:
            track.wav = None    # Purge track wavs to hopefully free memory
        
    def __repr__(self):
        return self.name
    
    def __str__(self):
        return self.__repr__()

    def description(self):
        s = f"{self.name}\n"
        for track in self.tracks:
            s = s + f"{track.string}: {track.song_name}\n"
        return s
        
    
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
    keys = []
    for track in tracks:
        if not track.pitchless:
            keys.append(track.key)
    if len(keys) <= 0:
        return 0
    return int(np.mean(keys))

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
    if track.pitchless:
        return track
    semitones = new_key - track.key % 12
    #print(f"Transposing {track} by {semitones} semitones to {new_key}...")
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

def merge_same_bpm_and_key(tracks):
    new_tracks = []
    groups = defaultdict(list)
    for track in tracks:
        groups[(track.bpm, track.key)].append(track)
    for group in groups.values():
        if len(group) <= 1:
            new_tracks.append(group[0])
            continue
        new_wav = add_wav_list([track.wav for track in group])
        new_tracks.append(Track("N/A (Merged Track)", group[0].bpm, group[0].key, group[0].mode, wav=new_wav))
    return new_tracks

# song_length: Fraction of 32 bars for the desired song length
# returns Mashup object
def mashup_tracks(tracks, new_bpm=None, new_key=None, song_length=0.5):
    if new_bpm is None:
        new_bpm = find_middle_bpm(tracks)
    if new_key is None:
        new_key = find_middle_key(tracks)

    original_tracks = tracks.copy()
    tracks = merge_same_bpm_and_key(tracks)
    transposed = []
    for track in tracks:
        track.wav = track.wav[:int(song_length*len(track.wav))]
        transposed.append(transpose_track(track, new_key))
    transposed = merge_same_bpm_and_key(transposed)

    transposed_stretched = []
    for track in transposed:
        transposed_stretched.append(stretch_track(track, new_bpm))

    mashup_wav = add_wav_list([track.wav for track in transposed_stretched])
    return Mashup(mashup_wav, original_tracks)

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

def load_track(file):
    directory = os.path.split(file)[0]
    with open(os.path.join(directory, "info.json")) as info_file:
        s = json.load(info_file)
    trackstring = get_track_string(file)
    trackType = globals()[trackstring]
    key, mode, pitchless = get_optional_values(s)
    return trackType(s["name"], s["bpm"], KEYS[key], mode, pitchless=pitchless, path=file)
    

def infinite_random_mashup(track_types = ["Drums", "Bass", "Other", "Vocals"], bpm=None, key=None, song_length = 0.5):
    mashup = random_mashup(track_types, bpm, key, song_length)
    while True:
        playObject = play_wav_array(mashup.wav, mashup.sr)
        print(f"Now playing: {mashup.description()}")
        mashup = random_mashup(track_types, bpm, key, song_length)
        # TODO eliminate the delay between tracks
        playObject.wait_done()

infinite_random_mashup(bpm = 110, key = KEYS["Db"], song_length=0.25)
'''

drums = load_track("songs/sanctuary/drums.wav")
other = load_track("songs/sanctuary/other.wav")
bass = load_track("songs/beat it/bass.wav")
vocals = load_track("songs/sanctuary/vocals.wav")
mashup = mashup_tracks([drums, other, bass, vocals])
playObject = play_wav_array(mashup.wav, mashup.sr)
playObject.wait_done()
'''