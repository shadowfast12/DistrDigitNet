package org.digitNet.client;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.digitNet.SerializationUtil;
import org.digitNet.server.MessageType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.net.Socket;
import java.util.List;

/**
 * Worker: handshake once, then loop pulling shards until done.
 */
public class WorkerClient {
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: WorkerClient <masterHost> <masterPort>");
            System.exit(1);
        }
        String host = args[0];
        int    port = Integer.parseInt(args[1]);

        try (Socket sock = new Socket(host, port);
             DataOutputStream out = new DataOutputStream(sock.getOutputStream());
             DataInputStream  in  = new DataInputStream(sock.getInputStream()))
        {
            // handshake: model JSON
            int cLen = in.readInt();
            byte[] cBuf = new byte[cLen];
            in.readFully(cBuf);
            MultiLayerConfiguration conf =
                    MultiLayerConfiguration.fromJson(new String(cBuf, "UTF-8"));
            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();

            // handshake: hyperparams
            int localEpochs = in.readInt();
            int batchSize   = in.readInt();
            System.out.printf("Handshake received: localEpochs=%d, batchSize=%d%n",
                    localEpochs, batchSize);

            // log loss every 10 minibatches
            model.setListeners(new ScoreIterationListener(10));

            // shard loop
            while (true) {
                int code = in.readInt();
                if (code == MessageType.NO_MORE_SHARDS.code) {
                    System.out.println("No more shards; exiting.");
                    break;
                }
                if (code != MessageType.SHARD_DATA.code) {
                    throw new RuntimeException("Unexpected code: " + code);
                }

                // receive global params
                int pLen = in.readInt();
                byte[] pBuf = new byte[pLen];
                in.readFully(pBuf);
                model.setParams(Nd4j.fromByteArray(pBuf));

                // receive features
                int xLen = in.readInt();
                byte[] xBuf = new byte[xLen]; in.readFully(xBuf);
                INDArray X = Nd4j.fromByteArray(xBuf);

                // receive labels
                int yLen = in.readInt();
                byte[] yBuf = new byte[yLen]; in.readFully(yBuf);
                INDArray Y = Nd4j.fromByteArray(yBuf);

                // local training
                List<DataSet> data = new DataSet(X, Y).asList();
                var iter = new ListDataSetIterator<>(data, batchSize);
                for (int e = 1; e <= localEpochs; e++) {
                    iter.reset();
                    model.fit(iter);
                    System.out.printf("  epoch %d/%d, loss=%.4f%n",
                            e, localEpochs, model.score());
                }

                // send back updated params
                byte[] upd = Nd4j.toByteArray(model.params());
                out.writeInt(upd.length);
                out.write(upd);
                out.flush();
                System.out.println("Sent updated parameters");
            }
        }
    }
}
