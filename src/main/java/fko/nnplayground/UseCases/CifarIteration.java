/*
 * MIT License
 *
 * Copyright (c) 2018 Frank Kopp
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

package fko.nnplayground.UseCases;

import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.nd4j.linalg.api.buffer.DataBuffer;
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

    // create the classifier
    LinearClassifier lc = new LinearClassifier(32, 32, 3, 10);

    lc.setUseRegularization(true);
    lc.setRegularizationLamda(0.1d);

//    lc.setLossFunction(LinearClassifier.LossFunction.SVM);
//    lc.train(dsTrain, 1);

    lc.setLossFunction(LinearClassifier.LossFunction.SOFTMAX);
    lc.train(dsTrain, 100);


  }
}