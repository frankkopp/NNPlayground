package fko.nnplayground.API;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public interface Network {

  void train(DataSetIterator dataSetIter, int epochs, int iterations);
  void train(DataSet dataSet, int epochs, int iterations);
  void train(INDArray features, INDArray labels, int epochs, int iterations);

}
