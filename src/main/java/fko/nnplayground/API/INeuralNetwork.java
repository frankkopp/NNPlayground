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

package fko.nnplayground.API;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.List;

/**
 * TODO: Javadoc
 */
public interface INeuralNetwork {

  void saveToFile(String nnSaveFile);

  void train(DataSetIterator dataSetIter, int epochs, int iterations);

  void train(DataSet dataSet, int epochs, int iterations);

  void train(INDArray features, INDArray labels, int epochs, int iterations);

  void eval(DataSetIterator dataSetIterator);

  void eval(DataSet dataSet);

  INDArray predict(INDArray features);

  void addLayer(ILayer layer);

  void addLayer(ILayer... layer);

  List<ILayer> getLayerList();

  int getInputLength();

  int getOutputLength();

  int getExamplesSeenTraining();

  int getExamplesSeenEval();

  double getCurrentScore();

  double getLearningRate();

  double getPrecision();

  double getRecall();

  double getAccuracy();

  double getF1score();

  void setLearningRate(double learningRate);

  void addListener(ITrainingListener listener);

  void addListener(ITrainingListener... listener);

  void removeListener(ITrainingListener listener);

}
