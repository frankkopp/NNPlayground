package fko.nnplayground.API;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public interface Network {

  void train(DataSetIterator dataSetIter, int epoches, int iterations, double learningRate);



}
