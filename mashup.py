import soundfile as sf
import pyrubberband as pyrb
import numpy as np
from numpy import ndarray
import os
import json
import pathlib
from collections import defaultdict
from collections.abc import Iterable
from warnings import warn
from multiprocessing.pool import ThreadPool
import re
import time
import msvcrt
from copy import deepcopy
from audioPlayer import AudioPlayer

# TODO: song transition code is very cursed and does not account for lag
# TODO: maybe make the song name attribute in info.json not forced unique by using some has as an ID
# TODO: Mode where the user can type which stems should stay, rest gets a new random stem
# TODO: move some functions into the Stem class
# TODO: key finding algo has a bug, check mashup between constellation and gohouteki or antonymph x the less i know
# TODO: use better stretching library?

NS_IN_ONE_SECOND = 1000000000
SONGDIR = "songs"
STEM_PATHS = [file for file in pathlib.Path(SONGDIR).rglob("*.[wW][aA][vV]")]
SONG_PATHS = [entry.path for entry in os.scandir(SONGDIR) if entry.is_dir()]
KEYS = {'A':0, 'Bb':1, 'B':2, 'C':3, 'Db':4, 'D':5, 'Eb':6, 'E':7, 'F':8, 'Gb':9, 'G':10, 'Ab': 11}
MODES = ('Minor', 'Major')
N_BEATS = 32*4 # Number of bars in each song. All songs provided in the songs folder should be trimmed to this number.

class Stem:
    """Class that holds the audio data and attributes of one stem of a song"""
    def __init__(self, song_name, bpm, key, mode, pitchless=["Drums"], wav: np.ndarray|None=None, sr=44100, path=None, halftime=False, doubletime=False, balance=0.5):
        self.song_name = song_name
        self.bpm = bpm
        self.key = key
        self.mode = mode.capitalize()
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
            wav, self.sr = sf.read(file=str(path), always_2d=True, dtype="float32")
            wav = balance * wav
        self.wav = wav

        try:
            ls = [s.capitalize() for s in pitchless]
            self.pitchless = self.string in ls
        except:
            self.pitchless = self.string == "Drums"

        if not self.pitchless and self.mode not in MODES:
            warn(f'\"{self.mode}\" is not a recognized mode. Stem \"{self.string}\" of \"{self.song_name}\" '+
                f'will will be treated as Minor.')
            self.mode = "Minor"
        
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
    
    def copy(self):
        return deepcopy(self)

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
    def __init__(self, wav, stems, bpm, sr=44100):
        self.wav = wav
        self.stems = stems
        self.name = make_mashup_name(stems)
        self.bpm = bpm
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
    
    def print_now_playing(self):
        print(f"Now playing:\t{self.description()}")
        
    
def sum_wav_list(wavs: list[ndarray]) -> ndarray:
    """Sum a list of audio arrays into one audio array.

    :param wavs: List of numpy arrays representing audio (shape: samples x channels).
    :return: A single numpy array which is the sample-wise sum of all inputs.
    """
    accu = wavs[0]
    if len(wavs) <= 1:
        return accu
    for wav in wavs[1:]:
        accu = sum_wavs(accu, wav)
    return accu
    
def stretch_stem(stem: Stem, new_bpm) -> Stem:
    """Stretch a stem to a new BPM.

    If the stem is marked as `doubletime` the audio will be cut in half before stretching.
    If the stem is marked as `halftime` the stretched audio will be repeated twice.

    :param stem: The input :class:`Stem` to stretch.
    :param new_bpm: Target beats-per-minute to stretch the stem to.
    :return: A new :class:`Stem` instance containing the stretched audio.
    """
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

def get_stem_type(stem: Stem) -> type:
    """Return the concrete Stem subclass type for a given stem.

    :param stem: A :class:`Stem` instance.
    :return: The class (type) corresponding to the stem's `string` attribute, or
             :class:`Stem` if no matching class is found in globals().
    """
    try:
        return globals()[stem.string]
    except:
        return Stem

def find_middle_bpm(stems: list[Stem], ignore_types=None):
    """Find the median BPM in a list of stems.

    The algorithm considers doubling or halving individual stem tempi to minimize
    the standard deviation across the selected tempos. By default `Bass` and
    `Other` types are ignored when computing the candidate BPMs.

    :param stems: List of :class:`Stem` instances to analyze.
    :param ignore_types: Optional list of Stem classes to ignore when computing.
    :return: The median BPM (float) selected for the group. Side-effects: may
             set `halftime` or `doubletime` flags on the provided stems.
    """
    if ignore_types is None:
        ignore_types = [Bass, Other]

    tempi = [stem.bpm for stem in stems if type(stem) not in ignore_types]
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
    return np.median(best_tempo_list)

def find_middle_key(stems: Iterable[Stem], ignore_types=None) -> int:
    """Find a key (0-11) that minimizes the maximum circular semitone distance
    to any non-pitchless stem in the provided list.

    :param stems: Iterable of :class:`Stem` instances to consider.
    :param ignore_types: Optional list of Stem classes to ignore (default ignores Bass).
    :return: Integer key in range 0-11 representing the chosen semitone key.
    """
    if ignore_types is None:
        ignore_types = [Bass]

    keys = [stem.key % 12 for stem in stems if not stem.pitchless and type(stem) not in ignore_types]
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
    """Sum two audio arrays, padding the shorter to the length of the longer.

    :param wav1: First audio numpy array (samples x channels).
    :param wav2: Second audio numpy array (samples x channels).
    :return: The element-wise sum of the two arrays, padded as necessary.
    """
    shorter = wav1
    longer = wav2
    if len(wav1) > len(wav2):
        shorter = wav2
        longer = wav1
    diff = len(longer) - len(shorter)
    if diff != 0:
        shorter = np.pad(shorter, ((0, diff), (0, 0)))
    return shorter + longer
    
def transpose_stem(stem: Stem, new_key: int) -> Stem:
    """Transpose a stem to a new key unless the stem is pitchless.

    The function chooses the shortest semitone direction (up or down) and
    performs a pitch shift using `pyrubberband`.

    :param stem: The :class:`Stem` to transpose.
    :param new_key: Target key as an integer (semitones, 0-11).
    :return: A new :class:`Stem` instance with audio pitch-shifted to `new_key`.
    """
    if stem.pitchless:
        return stem
    semitones = new_key - stem.key
    if semitones >= 7:
        semitones -= 12     # Pitch down if lower octave is closer 
    if semitones <= -6:
        semitones += 12     # Pitch up if upper octave is closer
    new_wav = pyrb.pitch_shift(stem.wav, stem.sr, semitones)
    stemType = get_stem_type(stem)
    return stemType(stem.song_name, stem.bpm, new_key, stem.mode, wav=new_wav, sr=stem.sr,
            halftime=stem.halftime, doubletime=stem.doubletime)

def merge_same_bpm_and_key(stems: Iterable[Stem]) -> list[Stem]:
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


def mashup_stems(stems: list[Stem], bpm=.0, key: int|None=None, start=0, end=1) -> Mashup:
    """Create a mashup from a list of stems.

    :param stems: List of :class:`Stem` objects to include in the mashup.
    :param bpm: Optional target BPM. If 0.0, the function computes a middle BPM.
    :param key: Optional target key (int). If None, the function computes a middle key.
    :param start: Fraction (0-1) of each stem to treat as the segment start.
    :param end: Fraction (0-1] of each stem to treat as the segment end.
    :return: A :class:`Mashup` instance containing the combined audio and metadata.
    """
    if bpm == .0:               bpm = find_middle_bpm(stems)
    if key is None:             key = find_middle_key(stems)
    if start >= 1 or end <= 0:  raise ValueError("Start point must be within [0, 1); end point must be within (0, 1])")
    if start >= end:            raise ValueError("Starting point must be before the end point")

    original_stems = stems.copy()
    stems = merge_same_bpm_and_key(stems)

    for stem in stems:
        stem.wav = stem.wav[int(start*len(stem.wav)) : int(end*len(stem.wav))]
    with ThreadPool() as pool:
        transposed = pool.starmap(transpose_stem, [(stem, key) for stem in stems])
    transposed = merge_same_bpm_and_key(transposed)

    with ThreadPool() as pool:
        transposed_stretched = pool.starmap(stretch_stem, [(stem, bpm) for stem in transposed])

    mashup_wav = sum_wav_list([stem.wav for stem in transposed_stretched])
    return Mashup(mashup_wav, original_stems, bpm)
    
def random_mashup(generate_stem_groups: list[list[str]]|None=None, predefined_stems: list[Stem]|None=None, vocalswap=False, **kwargs):
    """Generate a random mashup composed from random stems.

    :param generate_stem_groups: Nested list of stem type groups. Types in the same
                                 sublist will be picked from a single song. Default
                                 is `[["Drums"], ["Bass", "Other"], ["Vocals"]]`.
    :param predefined_stems: Optional list of fixed :class:`Stem` objects to include.
    :param vocalswap: If True, pick all instrumental sources from the same song.
    :param kwargs: Additional keyword arguments forwarded to :func:`mashup_stems`.
    :return: A :class:`Mashup` instance representing the generated mashup.
    """
    if generate_stem_groups is None: generate_stem_groups = [["Drums"], ["Bass", "Other"], ["Vocals"]]
    if predefined_stems is None:     predefined_stems = []
    if vocalswap:                    generate_stem_groups = [["Drums", "Bass", "Other"], ["Vocals"]]

    stems = [stem.copy() for stem in predefined_stems]
    for stem_types in generate_stem_groups:
        exclude_songs = [stem.song_name for stem in stems]
        stems.extend(load_random_stem_group(stem_types, exclude_songs))
    return mashup_stems(stems, **kwargs)

def get_random_instrumental_path() -> str:
    pattern = re.compile(r"(?i)(?<!vocals)\.wav$")
    for i in range(500):
        song_path = np.random.choice(SONG_PATHS)
        stem_files = [file for file in pathlib.Path(song_path).rglob("*") if pattern.search(str(file))]
        if stem_files: return song_path
    raise Exception(f"Could not find an instrumental stem after looking through {i+1} song folders")

def get_random_stem_path(stem_type: str) -> str:
    for i in range(500):
        song_path = np.random.choice(SONG_PATHS)
        stem_path = os.path.join(song_path, f"{stem_type}.wav")
        if os.path.exists(stem_path): return song_path
    raise Exception(f'Could not find a "{stem_type}" stem after looking through {i+1} song folders')

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
    """Return the stem name (string) extracted from a file path.

    Example: `/path/to/Drums.wav` -> `Drums`.

    :param file_path: Path to a stem file.
    :return: Capitalized stem name without extension.
    """
    return os.path.split(file_path)[1][:-4].capitalize()

def get_song_name(song_path):
    """Read the `info.json` for a song and return its declared name.

    :param song_path: Path to the song directory containing `info.json`.
    :return: The `name` field from the song's `info.json`.
    """
    with open(os.path.join(song_path, "info.json"), encoding='utf-8') as info_file:
        infos = json.load(info_file)
    return infos["name"]
    
def load_random_stem(stem_type: str|None = None):
    """Load a random stem file, optionally filtering by stem type.

    :param stem_type: Optional stem name to look for (e.g. `Drums`, `Vocals`).
                      If None, a random stem file is returned.
    :return: A :class:`Stem` instance loaded from a randomly chosen file.
    :raises Exception: If no matching stem of the requested type is found.
    """
    paths = STEM_PATHS.copy()
    np.random.shuffle(paths) #type: ignore
    if stem_type is None:
        return load_stem(paths[0])
    for path in paths:
        if get_stem_string(path).lower() == stem_type.lower():
            return load_stem(path)
    raise Exception(f'No stems of type "{stem_type}" were found')

def load_random_stem_group(stem_types: list[str]|None=None, exclude_songs: list[str]|None=None) -> list[Stem]:
    """
    Loads stems from a random song

    :param stem_types: Stem types to load. Not all are guaranteed to be present in the selected song, but at least one returned stem is guaranteed.
    :param exclude_songs: Song list to exclude from random selection
    :return: Stems loaded from a random song
    :rtype: list[Stem]
    """
    if stem_types is None or len(stem_types) == 0: stem_types = ["Drums", "Vocals", "Bass", "Other"]
    if exclude_songs is None: exclude_songs = []

    stems = []
    while len(stems) == 0:
        song_path = np.random.choice(SONG_PATHS)
        while get_song_name(song_path) in exclude_songs:
            song_path = np.random.choice(SONG_PATHS)
        stems = load_stem_group(song_path, stem_types)
    return stems

def load_stem_group(song_path: str, stem_types: list[str]|None=None) -> list[Stem]:
    """
    Loads stems from a song

    :param stem_types: Stem types to load. Not all are guaranteed to be present in the selected song.
    :return: Specified stems from the song
    :rtype: list[Stem]
    """
    if stem_types is None or len(stem_types) == 0: stem_types = ["Drums", "Vocals", "Bass", "Other"]

    stems = []
    for stem_type in stem_types:
        stem_path = os.path.join(song_path, f"{stem_type}.wav")
        if os.path.exists(stem_path):
            stems.append(load_stem(stem_path))
    return stems

def get_optional_values(song_dict: dict):
    try:
        key = KEYS[song_dict["key"]]
    except KeyError:
        key = None
        
    try:
        mode = song_dict["mode"]
    except KeyError:
        mode = ""

    try:
        pitchless = song_dict["pitchless"]
    except KeyError:
        pitchless = ["Drums"]

    return key, mode, pitchless

def load_stem(path: pathlib.Path|str, balance=0.5) -> Stem|None:
    """Load a stem from a file path using metadata from the song's `info.json`.

    :param path: Path to the stem WAV file.
    :param balance: Multiplier applied to the stem audio when loading (default 0.5).
    :return: Constructed :class:`Stem` instance (or subclass) or None on failure.
    """
    directory = os.path.split(path)[0]
    info_path = os.path.join(directory, "info.json")
    try:
        with open(info_path, encoding='utf-8') as info_file:
            infos = json.load(info_file)
    except Exception as e:
        warn(str(e))
        return
    stem_string = get_stem_string(path)
    stem_constructor = globals()[stem_string]
    key, mode, pitchless = get_optional_values(infos)
    return stem_constructor(infos["name"], infos["bpm"], key, mode, pitchless=pitchless, path=path, balance=balance)
    

def infinite_random_mashup(n_segments=4, **kwargs):
    """Play random mashups indefinitely.

    :param n_segments: Number of segments per mashup; each source is trimmed to
                       `1/n_segments` of its length for each segment.
    :param kwargs: Additional keyword arguments forwarded to :func:`random_mashup`.
    :return: This function runs indefinitely until interrupted.
    """
    print("Generating first mashup, please wait...\nPress 's' to skip a mashup segment\n")
    # TODO: prepare a queue of a couple of mashups with the thread pool, for smoother skipping
    player, track_duration = None, None
    for i in range(999_999_999):
        start = (i % n_segments) / n_segments
        end = ((i+1) % n_segments) / n_segments
        if end == 0: end = 1
        kwargs['start'] = start; kwargs['end'] = end
        with ThreadPool() as pool:
            result = pool.apply_async(random_mashup, kwds=kwargs)
            wait_or_skip(track_duration, player)
            mashup = result.get()
        player = AudioPlayer(mashup.wav, mashup.sr)
        player.start()
        mashup.print_now_playing()
        track_duration = N_BEATS / n_segments / mashup.bpm * 60
        

def play_mashup(mashup: Mashup, print_info = True):
    """Play a :class:`Mashup` using an `AudioPlayer`.

    :param mashup: The :class:`Mashup` instance to play.
    :param print_info: Whether to print now-playing information (default True).
    """
    if print_info: mashup.print_now_playing()
    player = AudioPlayer(mashup.wav, mashup.sr)
    player.start()
    player.join()

def wait_or_skip(track_duration, play_object):
    """Wait for a currently-playing track to finish, or skip if user presses `s`.

    :param track_duration: Duration of the track in seconds.
    :param play_object: Playback object with a `.stop()` method (or None).
    :return: `True` if the user skipped playback, `False` if playback completed.
    """
    if play_object is None: return False
    track_duration -= 0.04 # Don't ask me why this has to be here but it does
    start = time.perf_counter()
    while True:
        if msvcrt.kbhit():
            if msvcrt.getwch() == 's':
                print("Mashup segment skipped...")
                play_object.stop()
                return True
        elapsed = (time.perf_counter() - start)
        if elapsed >= track_duration:
            return False
        time.sleep(0.001)  # Sleep to reduce CPU usage

def find_song(name: str) -> str:
    """Find a song directory whose path matches `name`.

    :param name: Substring to match against file/directory names under `SONGDIR`.
    :return: Path to the first matching location as a string.
    :raises FileNotFoundError: If no match is found.
    """
    # TODO support search through song name metadata
    matches = list(pathlib.Path(SONGDIR).rglob(f"*{name}*"))
    if matches:
        return str(matches[0])
    else:
        raise FileNotFoundError(f"No song found matching '{name}'")

def find_stem(song_name: str, stem_type: str):
    """Return the filesystem path to a stem WAV file for a song.

    :param song_name: Name or substring to locate the song directory.
    :param stem_type: Stem filename (without extension), e.g. `drums`, `vocals`.
    :return: Path to the stem WAV file.
    """
    return os.path.join(find_song(song_name), stem_type + ".wav")

def quick_mashup(drums=None, bass=None, other=None, vocals=None, balances=[1]*4, **kwargs):
    """Quick helper to build and play a mashup from named stems.

    :param drums: Song name for drums stem (or None).
    :param bass: Song name for bass stem (or None).
    :param other: Song name for other stem (or None).
    :param vocals: Song name for vocals stem (or None).
    :param balances: List of per-stem balance multipliers.
    :param kwargs: Forwarded to :func:`mashup_stems`.
    """
    print(loading_message())
    stems = []
    for stem_str, stem_type, balance in zip([drums, bass, other, vocals], ["drums", "bass", "other", "vocals"], balances):
        if stem_str:
            stems.append(load_stem(find_stem(stem_str, stem_type), balance))
    play_mashup(mashup_stems(stems, **kwargs))

def quick_vocalswap_mashup(instr: str, acap: str, **kwargs):
    """Quick helper: take instrumentals from one song and vocals from another.

    :param instr: Song name providing instrumental stems.
    :param acap: Song name providing vocal stems (acapella).
    :param kwargs: Forwarded to :func:`mashup_stems`.
    """
    print(loading_message())
    stems =      load_stem_group(find_song(instr), stem_types=["Drums", "Bass", "Other"])
    stems.extend(load_stem_group(find_song(acap),  stem_types=["Vocals"]))
    play_mashup(mashup_stems(stems, **kwargs))

def loading_message():
    return "Preparing Mashup..."

def main():
    # quick_vocalswap_mashup("photo", "sebb rem")
    seed = np.random.randint(low=0, high=999999)
    np.random.seed(seed)
    print(f"Using seed {seed}")
    # infinite_random_mashup(vocalswap=True, n_segments=1)
    infinite_random_mashup(n_segments=4, 
                           predefined_stems=[load_stem(find_stem("monitor", "bass")), load_stem(find_stem("monitor", "other"))], 
                           generate_stem_groups=[["Drums"], ["Vocals"]],
                           bpm=132,
                           key=KEYS["D"])

if __name__ == "__main__":
    main()