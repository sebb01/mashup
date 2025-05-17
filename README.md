## Python Version
Because of an [issue in SimpleAudio](https://github.com/hamiltron/py-simple-audio/issues/74), only Python versions below 3.12 are supported. This library is tested with Python 3.10.6.

## How to import your own stems
- Make a folder inside of the `\songs` folder
- Drop stem(s) as .wav file(s) into this folder. Each stem should be named either:
    - `bass.wav`
    - `drums.wav`
    - `vocals.wav`
    - `other.wav`

    - The stem file can also have any other name, but then it will only be included in a mashup when the `track_types` argument is passed and the track name is included in that list
- Make an `info.json` file in this folder

### `info.json` file format
You can look at `info_template.json` to see how to format your file. The following fields must be defined:
- `name (string)`: Name of the song (must be unique!)
- `bpm (numerical)`: Tempo (BPM) of the song

I highly recommend to also define the following fields whenever applicable:
- `key (string)`: Key of the song, for example "Eb"
- `mode (string)`: Mode of the song; Currently only "Major" and "Minor" are supported
- `pitchless (list of strings)`: A list of all stems that should *not* get pitch shifted. If this field is not provided, the "Drums" track is considered pitchless by default. If your song has exclusively rap vocals with no sung parts, I recommend to include the "Vocals" track in this list.