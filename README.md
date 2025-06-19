**RUN Parameter Server:** 
```
java -jar DistrDigitNet-1.0-SNAPSHOT-jar-with-dependencies.jar <port> <learningRate> <localEpochs> <batchSize> <numShards><path/to/train-images.idx3-ubyte> <path/to/train-labels.idx1-ubyte>
```

EX: 
```
java -jar DistrDigitNet-1.0-SNAPSHOT-jar-with-dependencies.jar 5000 0.001 5 64 100 data/train-images.idx3-ubyte data/train-labels.idx1-ubyte

```
- port: TCP port for workers (e.g., 5000)
- learningRate: e.g., 0.001
- localEpochs: passes per shard on each worker (e.g., 5)
- batchSize: mini-batch size (e.g., 64)
- numShards: how many pieces to split the 60 000-image train set (e.g., 100)


**RUN Worker Clients**
```
java -cp DistrDigitNet-1.0-SNAPSHOT-jar-with-dependencies.jar org.digitNet.client.WorkerClient <serverHost> <port>
```

EX: 
```
java -cp DistrDigitNet-1.0-SNAPSHOT-jar-with-dependencies.jar org.digitNet.client.WorkerClient localhost 5000

```


**HOW IT WORKS:**
DataLoader reads full MNIST train set and splits it into numShards shards.

**ParameterServer**:

- Listens on port
- Handles each worker in its own thread:
- Sends model JSON & hyperparams
- Streams model.params(), shard features & labels
- Receives locally-updated parameters 

- Applies FedAvg update:
               ` theta_new = theta_old - lr * (theta_old - theta_worker);`
                 Continues until all shards are dispatched, then saves globalModel.zip.

**WorkerClient**:

- Connects once to server, receives model & hyperparams
- Loops reading SHARD_DATA:
- Receives global params + shard data
- Calls model.fit(...) for localEpochs epochs
- Sends updated model.params() back
- Exits on NO_MORE_SHARDS.


