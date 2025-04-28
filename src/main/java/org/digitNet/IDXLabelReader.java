package org.digitNet;

import java.io.*;

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;

public class IDXLabelReader {
    private INDArray labels; // shape: [numLabels, 10] one-hot

    public IDXLabelReader(String fileName) {
        File f = new File(fileName);
        if(!f.exists() || !f.canRead()){
            System.err.println("ERROR: FILE IS INVALID");
            return;
        }

        try(DataInputStream dis = new DataInputStream(new FileInputStream(f))) {
            int magic = dis.readInt();
            if (magic != 0x00000801) {
                System.err.println("Invalid IDX label file (magic: " + magic + ")");
            }

            int numLabels = dis.readInt();
            // Create one-hot labels: shape [numLabels, 10]
            labels = Nd4j.zeros(numLabels, 10);

            for (int i = 0; i < numLabels; i++) {
                int lbl = dis.readUnsignedByte();         // 0–9
                labels.putScalar(i, lbl, 1.0);            // set one-hot
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public INDArray getLabels() {
        return labels;
    }
}
