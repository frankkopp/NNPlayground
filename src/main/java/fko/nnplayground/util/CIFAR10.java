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

package fko.nnplayground.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * CIFAR10
 */
public class CIFAR10 {

  private static final Logger LOG = LoggerFactory.getLogger(CIFAR10.class);

  private static final String basePath = "./var/data" + "/cifar10";
  //private static final String dataUrl =  "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz";
  private static final String dataUrl =  "http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";

  public static void main(String[] args) throws Exception {

    LOG.info("Data load and vectorization using path {}", basePath);
    String localFilePath = basePath + "/cifar-10-binary.tar.gz";

    if (DataUtilities.downloadFile(dataUrl, localFilePath)) {
      LOG.info("Data downloaded from {}", dataUrl);
    }

    if (!new File(basePath + "/cifar-10-binary").exists()) {
      DataUtilities.extractTarGz(localFilePath, basePath);
    }

    LOG.info("Data extracted. Finished", basePath);
  }

}
