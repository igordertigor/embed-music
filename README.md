# Music embeddings

This is using the fma dataset:
```bash
cd data/raw/
wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
wget https://os.unil.cloud.switch.ch/fma/fma_small.zip

unzip fma_metadata.zip
unzip fma_small.zip
```

For loading mp3 files, you seem to need [ffmpeg](https://github.com/pytorch/audio/issues/2363) installed.
