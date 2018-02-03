package fko.nnplayground;

import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
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

    // determines what ND4j uses internally as precision for floating point numbers
    Nd4j.setDataType(DataBuffer.Type.FLOAT);

    CifarDataSetIterator cifarTrainDataSetIterator =
        new CifarDataSetIterator(
            batchSize, numSamples, new int[] {height, width, channels}, preProcessCifar, true);

    CifarDataSetIterator cifarTestDataSetIterator =
        new CifarDataSetIterator(
            1000, numSamples, new int[] {height, width, channels}, preProcessCifar, false);

    LOG.info("Number of total train examples: {}", cifarTrainDataSetIterator.totalExamples());
    LOG.info("Number of total test examples: {}", cifarTestDataSetIterator.totalExamples());

    // scaling the dataset to 0..1.0
    DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    scaler.fit(cifarTrainDataSetIterator);
    cifarTrainDataSetIterator.setPreProcessor(scaler);

//    CifarViewer cv = new CifarViewer();

    // get 5.000 (batch size) for this exercise
    DataSet dsTrain = cifarTrainDataSetIterator.next();
    DataSet dsTest = cifarTestDataSetIterator.next();
    dsTrain.setLabelNames(cifarTrainDataSetIterator.getLabels());
    dsTest.setLabelNames(cifarTestDataSetIterator.getLabels());

    LOG.info("Number of train examples: {}", dsTrain.numExamples());
    LOG.info("Number of test examples: {}", dsTest.numExamples());

    LOG.info("Labels Train: {}", dsTrain.getLabelNamesList());
    //LOG.info("Labels Test: {}", dsTest.getLabelNamesList());

    // create the classifier
    LinearClassifier lc = new LinearClassifier(32, 32, 3, 10);

    lc.setUseRegularization(false);
    lc.setLamda(0.1d);

    lc.setLossFunction(LinearClassifier.LossFunction.SVM);
    lc.train(dsTrain, 1);

    lc.setLossFunction(LinearClassifier.LossFunction.SOFTMAX);
    lc.train(dsTrain, 1);


//    INDArray losses = Nd4j.zeros(dsTrain.numExamples());
//    for (int i=0; i<dsTrain.numExamples(); i++) {
//      // one sample
//      DataSet sample = dsTrain.get(i);
//      LOG.debug("Sample class: {}", dsTrain.getLabelName(sample.outcome()));
//      //cv.showImage(sample.getFeatures());
//
//      INDArray flattened = Nd4j.toFlattened(sample.getFeatureMatrix());
//      INDArray scoreMatrix = lc.score(flattened.transpose());
//      LOG.debug("Predicted class: {} (idx: {})",
//              dsTrain.getLabelName(scoreMatrix.argMax(0).getInt(0)),
//              scoreMatrix.argMax(0));
//
//      final double loss = lc.lossSVM(scoreMatrix, sample.outcome());
//      losses.put(0, i, loss);
//      LOG.debug("Loss: {}", loss);
//    }
//    LOG.debug("Overall lossSVM: {}", losses.meanNumber());

//    cv.showImage(dsTrain.get(0).getFeatures());

//    LOG.debug(
//        "Weight matrix: \n{} \nshape:\n{}",
//        lc.getWeightsMatrix(),
//        lc.getWeightsMatrix().shapeInfoToString());
//
//    INDArray row = lc.getWeightsMatrix().get(NDArrayIndex.interval(0,1)).reshape(1, 3, 32, 32);

//    LOG.debug(
//            "Row matrix: \n{} \nshape:\n{}",
//            row,
//            row.shapeInfoToString());

//    cv.showImage(row);

  }
}
