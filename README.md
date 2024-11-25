# Room mapping draft with Fasttext

## Install

```
pip install flask fasttext scipy
```

Then, download [cc.en.300.bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz) and put it in the same directory as the server and unzip it.

## Run

```
python3 server.py
```

Test with:

```
curl -X POST -H "Content-Type: application/json" -d @samples/input.json http://localhost:5555/
```

---

## Docker run

```
docker build -t room-mapping .
```