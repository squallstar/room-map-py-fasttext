# Room mapping

## Install

```
pip install flask fasttext scipy sentence-transformers datasets 'accelerate>=0.26.0'
```

## Run with the base Fasttext model

Download [cc.en.300.bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz) and put it in the same directory as the server and unzip it.

```
python3 server.py
```

Test with:

```
curl -X POST -H "Content-Type: application/json" -d @samples/input.json http://localhost:5555/
```

---

### Docker run

```
docker build -t room-mapping .
```

---

# Run with a trained model

## Train the model

Put the [room_names.csv](https://nuiteetravel.slack.com/files/U05E5Q1CBDY/F082287QKP1/4000000.zip) file in the same directory as the training scripts.

```
python3 train_prep_data.py
python3 train_model.py
```

Then run the server and use the `/trained` route when making requests.