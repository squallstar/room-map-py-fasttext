# Room mapping

## Install

```
pip install flask fasttext scipy sentence-transformers datasets 'accelerate>=0.26.0'
```

## Train the model

Put the [room_names.csv](https://nuiteetravel.slack.com/files/U05E5Q1CBDY/F082287QKP1/4000000.zip) file in the same directory as the training scripts.

```
python3 1_prepare.py
python3 2_train.py
python3 3_test.py
```

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
