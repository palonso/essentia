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

#include "tensorflowinputsicnn.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* TensorflowInputSiCNN::name = "TensorflowInputSiCNN";
const char* TensorflowInputSiCNN::category = "Spectral";
const char* TensorflowInputSiCNN::description = DOC(
  "This algorithm computes mel-bands specific to the input of SiCNN-based models.\n"
  "\n"
  "References:\n"
  "  [1] "
  "  [2] "
);


void TensorflowInputSiCNN::configure() {
  _windowing->configure("normalized", _normalizedWindowing,
                        "type", _windoyType,
                        "zeroPadding", _fftSize - _frameSize);

  _spectrum->configure("size", _fftSize);

  _melBands->configure("inputSize", _fftSize / 2 + 1,
                       "numberBands", _numberBands,
                       "sampleRate", _sampleRate,
                       "highFrequencyBound", _highFrequencyBound,
                       "lowFrequencyBound", _lowFrequencyBound,
                       "warpingFormula", _warpingFormula,
                       "weighting", _weighting,
                       "normalize", _normalize,
                       "type", _melType);

  _shift->configure("shift", _eps);

  _compression->configure("type", _compType);

  // Set the intermediate buffers.
  _windowing->output("frame").set(_windowedFrame);

  _spectrum->input("frame").set(_windowedFrame);
  _spectrum->output("spectrum").set(_spectrumFrame);

  _melBands->input("spectrum").set(_spectrumFrame);
  _melBands->output("bands").set(_melBandsFrame);

  _shift->input("array").set(_melBandsFrame);
  _shift->output("array").set(_shiftedFrame);

  _compression->input("array").set(_shiftedFrame);
}


void TensorflowInputSiCNN::compute() {
  const std::vector<Real>& frame = _frame.get();

  if ((int)frame.size() != _frameSize) {
    throw(EssentiaException("TensorflowInputSiCNN: This algorithm only accepts input frames of size 512."));
  }

  _windowing->input("frame").set(frame);
  _compression->output("array").set(_bands.get());
  // _spectrum->output("spectrum").set(_bands.get());

  _windowing->compute();
  _spectrum->compute();
  _melBands->compute();
  _shift->compute();
  _compression->compute();
}
