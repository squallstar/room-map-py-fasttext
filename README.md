# Room mapping

## Install

```
pip install flask fasttext
```

Then, download the fasttext model [cc.en.300.bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz) and unzip it in the base folder.

## Run the server

```
python3 server.py
```

### Test

```
curl -X POST -H "Content-Type: application/json" -d @samples/input.json http://localhost:5555/
```

---

### Docker run

```
docker build -t room-mapping .
```
