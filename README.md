## Import your own stems
- Make a folder in the `\songs` folder
- Drop stem(s) as .wav file(s) into this folder. Each stem should be named either:
    - `bass.wav`
    - `drums.wav`
    - `other.wav`
    - `vocals.wav`
- Make an `info.json` file in this folder

### `info.json` file format
You can look at `info_template.json` to see how to format your file. The following fields must be defined:
- `name (string)`: Name of the track
- `bpm (numerical)`: Tempo (BPM) of the track

I highly recommend to also define the following fields whenever applicable:
- `key (string)`: Key of the song, for example "Eb"
- `mode (string)`: Mode of the song; Currently only "Major" and "Minor" are supported
- `pitchless (list of strings)`: A list of all stems that should *not* get pitch shifted. If this field is not provided, the "Drums" track is considered pitchless by default. If your song has exclusively rap vocals with no sung parts, I recommend to include the "Vocals" track in this list.