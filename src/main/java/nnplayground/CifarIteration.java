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
  private static int batchSize = 100;
  private static boolean preProcessCifar = false; // use Zagoruyko's preprocess for Cifar

  public static void main(String[] args) throws Exception {

    CifarDataSetIterator cifarDataSetIterator =
        new CifarDataSetIterator(
            batchSize, numSamples, new int[] {height, width, channels}, preProcessCifar, true);

    LOG.info("Number of examples: {}", cifarDataSetIterator.totalExamples());
    LOG.info("Labels: {}", cifarDataSetIterator.getLabels());

    // scaling the dataset to 0..1.0
    DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    scaler.fit(cifarDataSetIterator);
    cifarDataSetIterator.setPreProcessor(scaler);

    CifarViewer cv = new CifarViewer();

    while (cifarDataSetIterator.hasNext()) {
      DataSet ds = cifarDataSetIterator.next();
      ds.forEach((ex) -> cv.showImage(ex));
    }
  }
}
