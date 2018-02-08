package fko.nnplayground.MinstNN;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SimpleNeuralNetworkTest {

  @BeforeEach
  void setUp() {}

  @Test
  void constructor() {
    SimpleNeuralNetwork snn = new SimpleNeuralNetwork(28,28,1,10,100,1234);

  }
}