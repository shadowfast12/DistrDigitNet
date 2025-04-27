package org.digitNet;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;

public class IDXFileReader {

    private INDArray features; // ND4j Array that holds all the images and each row represents one image, flattened to 1d

    /**
     * THIS PARSES THROUGH THE IDX FILE AND STORES THE IMAGES INTO THE IND ARRAY
     * @param fileName Name of the IDX File
     */
    public IDXFileReader(String fileName){
        File file = new File(fileName);

        if(!file.exists() || !file.canRead()){
            System.err.println("ERROR: FILE IS INVALID");
            return;
        }

        try(DataInputStream dis = new DataInputStream(new FileInputStream(file))){
            //READ IDX HEADER INFORMATION
            int magicNumber = dis.readInt(); // Should be 2051 for images
            int numImages = dis.readInt();
            int numRows = dis.readInt();
            int numCols = dis.readInt();

            System.out.println("Magic Number: " + magicNumber);
            System.out.println("Number of Images: " + numImages);
            System.out.println("Image Dimensions: " + numRows + " x " + numCols);

            int imageSize = numRows * numCols;
            features = Nd4j.create(numImages, imageSize);

            byte[] buffer = new byte[imageSize];
            for(int i = 0; i < numImages; i++){
                dis.readFully(buffer);

                double[] pixels = new double[imageSize];
                for(int j = 0; j < imageSize; j++){
                    pixels[j] = (buffer[j] & 0xFF) / 255.0; // CONVERT THE UNSIGNED BYTE TO DOUBLE (0.0 - 1.0)
                }
                features.putRow(i,Nd4j.create(pixels)); // PLACE NORMALIZED PIXEL VALUES INTO features INDArray
            }

        }catch(IOException e){
            System.err.println("ERROR: PARSING THROUGH FILE " + e.getMessage());
        }

    }

    /**
     * GETTER FOR THE IND ARRAY CONTAINING ALL THE IMAGES
     * @return INDArray
     */
    public INDArray getFeatures() {
        return features;
    }
}
