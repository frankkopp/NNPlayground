package fko.nnplayground.API;

import fko.nnplayground.nn.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.List;

public interface Network {

  void saveToFile(String nnSaveFile);

  void loadFromFile(String nnSaveFile);

  abstract List<ILayer> getLayerList();

  abstract void addLayer(Layer layer);

  void train(DataSetIterator dataSetIter, int epochs, int iterations);

  void train(DataSet dataSet, int epochs, int iterations);
  void train(INDArray features, INDArray labels, int epochs, int iterations);

  INDArray predict(INDArray features);

  abstract double getLearningRate();

  abstract void setLearningRate(double learningRate);

  void eval(DataSet dataSet);
}
