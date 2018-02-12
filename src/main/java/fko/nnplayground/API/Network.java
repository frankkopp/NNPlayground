package fko.nnplayground.API;

import fko.nnplayground.MinstNN.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.List;

public interface Network {

  /**
   * @return order list of layers - highest index is output layer
   */
  List<ILayer> getLayerList();

  /**
   * adds a new layer at the end of the layer list
   * @param layer
   */
  void addLayer(Layer layer);

  void train(DataSetIterator dataSetIter, int epochs, int iterations);
  void train(DataSet dataSet, int epochs, int iterations);
  void train(INDArray features, INDArray labels, int epochs, int iterations);

  double getLearningRate();

  void setLearningRate(double learningRate);
}
