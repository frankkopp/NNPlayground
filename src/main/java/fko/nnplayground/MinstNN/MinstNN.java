package fko.nnplayground.MinstNN;

import fko.nnplayground.util.DataUtilities;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;

/** MinstNN */
public class MinstNN {

  private static final Logger LOG = LoggerFactory.getLogger(MinstNN.class);

  private static final String basePath = "./var/data" + "/mnist";
  private static final String dataUrl =
      "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz";

  public static void main(String[] args) throws Exception {

    int height = 28;
    int width = 28;
    int channels = 1; // single channel for grayscale images
    int outputNum = 10; // 10 digits classification
    int batchSize = 3;

    int seed = 1234;
    Random randNumGen = new Random(seed);

    LOG.info("Data load and vectorization...");
    String localFilePath = basePath + "/mnist_png.tar.gz";

    if (DataUtilities.downloadFile(dataUrl, localFilePath)) {
      LOG.debug("Data downloaded from {}", dataUrl);
    }
    if (!new File(basePath + "/mnist_png").exists()) {
      DataUtilities.extractTarGz(localFilePath, basePath);
    }

    // vectorization of train data
    File trainData = new File(basePath + "/mnist_png/testing"); // TODO changed for debugging
    LOG.debug("Preparing training data...{}", trainData);
    FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
    ParentPathLabelGenerator labelMaker =
        new ParentPathLabelGenerator(); // parent path as the image label
    ImageRecordReader trainRR = new ImageRecordReader(height, width, channels, labelMaker);
    trainRR.initialize(trainSplit);
    DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum);

    // pixel values from 0-255 to 0-1 (min-max scaling)
    DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    scaler.fit(trainIter);
    trainIter.setPreProcessor(scaler);

    // vectorization of test data
    File testData = new File(basePath + "/mnist_png/testing");
    LOG.debug("Preparing test data...{}", testData);
    FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
    ImageRecordReader testRR = new ImageRecordReader(height, width, channels, labelMaker);
    testRR.initialize(testSplit);
    DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);
    testIter.setPreProcessor(scaler); // same normalization for better results

    final int sizeHiddenLayer = 5;
    SimpleNeuralNetwork snn = new SimpleNeuralNetwork(height, width, channels, outputNum, sizeHiddenLayer, seed);

    int nEpochs = 1;
    int iterations = 1;
    double learningRate = 0.1d;

    snn.train(trainIter, nEpochs, iterations, learningRate);


  }
}
