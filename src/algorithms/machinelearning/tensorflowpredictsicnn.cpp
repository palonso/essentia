/*
 * Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of Essentia
 *
 * Essentia is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License as published by the Free
 * Software Foundation (FSF), either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the Affero GNU General Public License
 * version 3 along with this program.  If not, see http://www.gnu.org/licenses/
 */

#include "tensorflowpredictsicnn.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* TensorflowPredictSiCNN::name = essentia::standard::TensorflowPredictSiCNN::name;
const char* TensorflowPredictSiCNN::category = essentia::standard::TensorflowPredictSiCNN::category;
const char* TensorflowPredictSiCNN::description = essentia::standard::TensorflowPredictSiCNN::description;


TensorflowPredictSiCNN::TensorflowPredictSiCNN() : AlgorithmComposite(),
    _frameCutter(0), _tensorflowInputSiCNN(0), _vectorRealToTensor(0), _tensorTranspose(0),
    _tensorToPool(0), _tensorflowPredict(0), _poolToTensor(0), _tensorToVectorReal(0),
    _configured(false) {

  declareInput(_signal, 4096, "signal", "the input audio signal sampled at 16 kHz");
  declareOutput(_predictions, 0, "predictions", "the output values from the model node named after `output`");
}


void TensorflowPredictSiCNN::createInnerNetwork() {
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  _frameCutter            = factory.create("FrameCutter");
  _tensorflowInputSiCNN   = factory.create("TensorflowInputSiCNN");
  _vectorRealToTensor     = factory.create("VectorRealToTensor");
  _tensorTranspose        = factory.create("TensorTranspose");
  _tensorToPool           = factory.create("TensorToPool");
  _tensorflowPredict      = factory.create("TensorflowPredict");
  _poolToTensor           = factory.create("PoolToTensor");
  _tensorToVectorReal     = factory.create("TensorToVectorReal");

  _tensorflowInputSiCNN->output("bands").setBufferType(BufferUsage::forMultipleFrames);
  
  _signal                                  >> _frameCutter->input("signal");
  _frameCutter->output("frame")            >> _tensorflowInputSiCNN->input("frame");
  _tensorflowInputSiCNN->output("bands")   >> _vectorRealToTensor->input("frame");
  _vectorRealToTensor->output("tensor")    >> _tensorTranspose->input("tensor");
  _tensorTranspose->output("tensor")       >> _tensorToPool->input("tensor");
  _tensorToPool->output("pool")            >> _tensorflowPredict->input("poolIn");
  _tensorflowPredict->output("poolOut")    >> _poolToTensor->input("pool");
  _poolToTensor->output("tensor")          >> _tensorToVectorReal->input("tensor");

  attach(_tensorToVectorReal->output("frame"), _predictions);

  _network = new scheduler::Network(_frameCutter);
}


void TensorflowPredictSiCNN::clearAlgos() {
  if (!_configured) return;
  delete _network;
}


TensorflowPredictSiCNN::~TensorflowPredictSiCNN() {
  clearAlgos();
}


void TensorflowPredictSiCNN::reset() {
  AlgorithmComposite::reset();
}


void TensorflowPredictSiCNN::configure() {
  if (_configured) {
    clearAlgos();
  }

  createInnerNetwork();

  int patchHopSize = parameter("patchHopSize").toInt();
  string lastPatchMode = parameter("lastPatchMode").toString();
  bool batchSize = parameter("batchSize").toInt();

  vector<int> inputShape({batchSize, 1, _patchSize, _numberBands});

  _frameCutter->configure("frameSize", _frameSize, "hopSize", _hopSize);

  _vectorRealToTensor->configure("shape", inputShape,
                                 "lastPatchMode", lastPatchMode,
                                 "patchHopSize", patchHopSize);

  // Swap the Channel axis as this family of models expect data shaped as
  // {Batch, Time, Mels, Channel} (BCHW -> BHWC).
  vector<int> permutation({0, 2, 3, 1});
  _tensorTranspose->configure("permutation", permutation);

  _configured = true;

  string input = parameter("input").toString();
  string output = parameter("output").toString();
  string isTrainingName = parameter("isTrainingName").toString();

  _tensorToPool->configure("namespace", input);

  _poolToTensor->configure("namespace", output);

  string graphFilename = parameter("graphFilename").toString();
  string savedModel = parameter("savedModel").toString();

  _tensorflowPredict->configure("graphFilename", graphFilename,
                                "savedModel", savedModel,
                                "inputs", vector<string>({input}),
                                "outputs", vector<string>({output}),
                                "isTrainingName", isTrainingName,
                                "squeeze", false);
}

} // namespace streaming
} // namespace essentia



namespace essentia {
namespace standard {

const char* TensorflowPredictSiCNN::name = "TensorflowPredictSiCNN";
const char* TensorflowPredictSiCNN::category = "Machine Learning";
const char* TensorflowPredictSiCNN::description = DOC(
  "This algorithm makes predictions using SiCNN-based models.\n"
  "\n"
  "It feeds 101 mel-spectrogram patches to the model and jumps a constant amount of frames "
  "determined by `patchHopSize`.\n"
  "The `batchSize` parameter parallelizes inference when a GPU and the CUDA libraries are "
  "available. When it is set to a value of -1 the algorithm runs a single TensorFlow session "
  "at the end of the audio stream, which can be memory exhausting for long files.\n"
  "\n"
  "To operate correctly, the algorithm requires this pipeline::\n"
  "\n"
  "  MonoLoader(sampleRate=22050) >> TensorflowPredictSiCNN\n"
  "\n"
  "Note: This algorithm does not make any check on the input model so it is "
  "the user's responsibility to make sure it is a valid one.\n"
  "\n"
  "References:\n"
  "\n"
  "  [1]  \n"
  "  [2]  "
);


TensorflowPredictSiCNN::TensorflowPredictSiCNN() {
    declareInput(_signal, "signal", "the input audio signal sampled at 16 kHz");
    declareOutput(_predictions, "predictions", "the output values from the model node named after `output`");

    createInnerNetwork();
  }


TensorflowPredictSiCNN::~TensorflowPredictSiCNN() {
  delete _network;
}


void TensorflowPredictSiCNN::createInnerNetwork() {
  _tensorflowPredictSiCNN = streaming::AlgorithmFactory::create("TensorflowPredictSiCNN");
  _vectorInput = new streaming::VectorInput<Real>();

  *_vectorInput  >> _tensorflowPredictSiCNN->input("signal");
  _tensorflowPredictSiCNN->output("predictions") >>  PC(_pool, "predictions");

  _network = new scheduler::Network(_vectorInput);
}


void TensorflowPredictSiCNN::configure() {
  // if no file has been specified, do not do anything
  _tensorflowPredictSiCNN->configure(INHERIT("graphFilename"),
                                     INHERIT("savedModel"),
                                     INHERIT("input"),
                                     INHERIT("output"),
                                     INHERIT("isTrainingName"),
                                     INHERIT("patchHopSize"),
                                     INHERIT("batchSize"),
                                     INHERIT("lastPatchMode"));
}


void TensorflowPredictSiCNN::compute() {
  const vector<Real>& signal = _signal.get();
  vector<vector<Real> >& predictions = _predictions.get();

  if (!signal.size()) {
    throw EssentiaException("TensorflowPredictSiCNN: empty input signal");
  }

  _vectorInput->setVector(&signal);

  _network->run();

  try {
    predictions = _pool.value<vector<vector<Real> > >("predictions");
  }
  catch (EssentiaException&) {
    predictions.clear();
  }

  reset();
}


void TensorflowPredictSiCNN::reset() {
  _network->reset();
  _pool.remove("predictions");
}

} // namespace standard
} // namespace essentia
