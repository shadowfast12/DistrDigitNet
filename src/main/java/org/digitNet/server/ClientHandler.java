package org.digitNet.server;

import org.digitNet.DataShard;
import org.digitNet.SerializationUtil;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.Socket;
import java.nio.charset.StandardCharsets;

/**
 * Handles one worker connection:
 *  1) Send model JSON + training hyperparams
 *  2) Loop: serve a shard + params → receive avg-gradient → apply it → repeat
 */
public class ClientHandler implements Runnable {
    private final Socket           socket;
    private final ShardManager     shardManager;
    private final MultiLayerNetwork model;
    private final double           learningRate;
    private final int              localEpochs;
    private final int              batchSize;

    public ClientHandler(
            Socket socket,
            ShardManager shardManager,
            MultiLayerNetwork model,
            double learningRate,
            int localEpochs,
            int batchSize
    ) {
        this.socket        = socket;
        this.shardManager  = shardManager;
        this.model         = model;
        this.learningRate  = learningRate;
        this.localEpochs   = localEpochs;
        this.batchSize     = batchSize;
    }

    @Override
    public void run() {
        try (DataOutputStream out = new DataOutputStream(socket.getOutputStream());
             DataInputStream  in  = new DataInputStream(socket.getInputStream()))
        {
            // 1) Send network configuration JSON
            byte[] js = model.getLayerWiseConfigurations()
                    .toJson()
                    .getBytes(StandardCharsets.UTF_8);
            out.writeInt(js.length);
            out.write(js);

            // 1b) Send hyperparameters
            out.writeInt(localEpochs);
            out.writeInt(batchSize);
            out.flush();

            // 2) Work loop
            while (true) {
                DataShard shard = shardManager.nextShard();
                if (shard == null) {
                    // signal no more work
                    out.writeInt(MessageType.NO_MORE_SHARDS.code);
                    out.flush();
                    break;
                }

                // notify client a shard is coming
                out.writeInt(MessageType.SHARD_DATA.code);

                // send features
                sendArray(shard.getFeatures(), out, false);
                // send labels
                sendArray(shard.getLabels(),   out, false);
                // send current global params
                sendArray(model.params(),      out, false);
                out.flush();  // one flush for all three

                // receive averaged gradient back
                int gLen = in.readInt();
                byte[] gBuf = new byte[gLen];
                in.readFully(gBuf);
                INDArray avgGrad = SerializationUtil.fromBytes(gBuf);

                // apply update: theta ← theta − lr * avgGrad
                synchronized (model) {
                    model.params().subi(avgGrad.mul(learningRate));
                }
            }
        } catch (IOException e) {
            System.err.println("ClientHandler error: " + e.getMessage());
        }
    }

    /**
     * Write INDArray as [length:int][bytes] to the stream.
     * @param arr          the INDArray to send
     * @param out          the DataOutputStream
     * @param flushAfter   whether to flush immediately
     */
    private void sendArray(INDArray arr, DataOutputStream out, boolean flushAfter)
            throws IOException {
        byte[] raw = SerializationUtil.toBytes(arr);
        out.writeInt(raw.length);
        out.write(raw);
        if (flushAfter) out.flush();
    }
}
