package org.digitNet;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

public class SerializationUtil {
    public static byte[] toBytes(INDArray arr) {
        try {
            return Nd4j.toByteArray(arr);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    public static INDArray fromBytes(byte[] raw) {
        return Nd4j.fromByteArray(raw);
    }
}
