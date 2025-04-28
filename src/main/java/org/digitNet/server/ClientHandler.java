package org.digitNet.server;

import org.digitNet.DataShard;
import org.digitNet.SerializationUtil;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicInteger;

public class ClientHandler implements Runnable {
    private final Socket socket;
    private final ShardManager shardManager;
    private final MultiLayerNetwork model;
    private final double learningRate;
    private final int localEpochs, batchSize, totalShards;
    private final AtomicInteger shardsDone;

    public ClientHandler(
            Socket socket,
            ShardManager shardManager,
            MultiLayerNetwork model,
            double learningRate,
            int localEpochs,
            int batchSize,
            AtomicInteger shardsDone,
            int totalShards
    ) {
        this.socket = socket;
        this.shardManager = shardManager;
        this.model = model;
        this.learningRate = learningRate;
        this.localEpochs = localEpochs;
        this.batchSize = batchSize;
        this.shardsDone = shardsDone;
        this.totalShards = totalShards;
    }

    @Override
    public void run() {
        try (DataOutputStream out = new DataOutputStream(socket.getOutputStream());
             DataInputStream  in  = new DataInputStream(socket.getInputStream()))
        {
            // Handshake: send model JSON
            byte[] js = model.getLayerWiseConfigurations().toJson()
                    .getBytes(StandardCharsets.UTF_8);
            out.writeInt(js.length);
            out.write(js);
            // send hyperparams
            out.writeInt(localEpochs);
            out.writeInt(batchSize);
            out.flush();

            // Loop: serve shards
            while (true) {
                DataShard shard = shardManager.nextShard();
                if (shard == null) {
                    out.writeInt(MessageType.NO_MORE_SHARDS.code);
                    out.flush();
                    break;
                }
                int done = shardsDone.incrementAndGet();
                System.out.printf("→ [%d/%d] dispatching shard%n", done, totalShards);

                // tell client
                out.writeInt(MessageType.SHARD_DATA.code);
                // send params, features, labels
                sendArray(model.params(),        out, false);
                sendArray(shard.getFeatures(),   out, false);
                sendArray(shard.getLabels(),     out, false);
                out.flush();

                // receive updated params
                int len = in.readInt();
                byte[] buf = new byte[len];
                in.readFully(buf);
                INDArray updated = SerializationUtil.fromBytes(buf);

                // apply federated update θ ← θ − η(θ − θ_client)
                synchronized(model) {
                    INDArray delta = model.params().sub(updated).mul(learningRate);
                    model.params().subi(delta);
                }
            }
        } catch (Exception e) {
            System.err.println("ClientHandler error: " + e.getMessage());
        }
    }

    private void sendArray(INDArray arr,
                           DataOutputStream out,
                           boolean flush) throws Exception {
        byte[] raw = SerializationUtil.toBytes(arr);
        out.writeInt(raw.length);
        out.write(raw);
        if (flush) out.flush();
    }
}
