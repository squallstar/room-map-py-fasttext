# Room mapping draft with Fasttext

## Install

```
pip install flask fasttext scipy
```

Then, download [cc.en.300.bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz) and put it in the same directory as the server and unzip it.

## Run

```
curl -X POST -H "Content-Type: application/json" -d @input.json http://localhost:5555/
```