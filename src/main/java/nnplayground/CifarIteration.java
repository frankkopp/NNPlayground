package nnplayground;

import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** CifarIteration */
public class CifarIteration {

  private static final Logger LOG = LoggerFactory.getLogger(CifarIteration.class);

  private static final String basePath = "./var/data" + "/cifar10";

  private static int height = 32;
  private static int width = 32;
  private static int channels = 3;
  private static int numSamples = 50000;
  private static int batchSize = 5000;
  private static boolean preProcessCifar = false; // use Zagoruyko's preprocess for Cifar

  public static void main(String[] args) throws Exception {

    CifarDataSetIterator cifarTrainDataSetIterator =
        new CifarDataSetIterator(
            batchSize, numSamples, new int[] {height, width, channels}, preProcessCifar, true);

    CifarDataSetIterator cifarTestDataSetIterator =
        new CifarDataSetIterator(
            batchSize, numSamples, new int[] {height, width, channels}, preProcessCifar, false);

    LOG.info("Number of total train examples: {}", cifarTrainDataSetIterator.totalExamples());
    LOG.info("Number of total test examples: {}", cifarTestDataSetIterator.totalExamples());

    // scaling the dataset to 0..1.0
    DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    scaler.fit(cifarTrainDataSetIterator);
    cifarTrainDataSetIterator.setPreProcessor(scaler);

    CifarViewer cv = new CifarViewer();

    // get 5.000 (batch size) for this exercise
    DataSet dsTrain = cifarTrainDataSetIterator.next();
    DataSet dsTest = cifarTestDataSetIterator.next();
    dsTrain.setLabelNames(cifarTrainDataSetIterator.getLabels());
    dsTest.setLabelNames(cifarTestDataSetIterator.getLabels());

    LOG.info("Number of train examples: {}", dsTrain.numExamples());
    LOG.info("Number of test examples: {}", dsTest.numExamples());

    LOG.info("Labels Train: {}", dsTrain.getLabelNamesList());
    LOG.info("Labels Test: {}", dsTest.getLabelNamesList());



    dsTrain.forEach(
        (ex) -> {
          cv.showImage(ex);
        });

  }
}
