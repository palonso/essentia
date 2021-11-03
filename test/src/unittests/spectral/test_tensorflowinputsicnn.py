#!/usr/bin/env python

# Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
#
# This file is part of Essentia
#
# Essentia is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation (FSF), either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the Affero GNU General Public License
# version 3 along with this program. If not, see http://www.gnu.org/licenses/


from essentia_test import *

FRAME_SIZE = 660
HOP_SIZE = 220
NBANDS = 96
SAMPLE_RATTE = 22050
EPS = 1.1e-8


class TestTensorflowInputSiCNN(TestCase):
    def testZero(self):
        # This mel-spectrogram implementation features a custom EPS value.
        # Assert that an empty input results on the correspondent log value.
        expected = numpy.log10(EPS * ones(NBANDS))
        self.assertEqualVector(TensorflowInputSiCNN()(zeros(FRAME_SIZE)), expected)

    def testRegression(self):
        expected = numpy.load(join(filedir(), 'tensorflowinputsicnn', 'expected_melbands.npy'))

        audio = MonoLoader(
            filename=join(testdata.audio_dir, "recorded/dubstep.wav"),
            sampleRate=SAMPLE_RATTE,
        )()

        tensorflowInputSiCNN = TensorflowInputSiCNN()
        frames = [
            tensorflowInputSiCNN(frame)
            for frame in FrameGenerator(audio, frameSize=FRAME_SIZE, hopSize=HOP_SIZE)
        ]
        obtained = numpy.mean(array(frames), axis=0)

        # We use fixed precision because in this implementation many mel bands are very close
        # to 0, making the relative differences very high.
        self.assertAlmostEqualVectorFixedPrecision(obtained, expected, 2)

    def testInvalidInput(self):
        self.assertComputeFails(TensorflowInputSiCNN(), [])

    def testWrongInputSize(self):
        # mel bands should fail for input size different to FRAME_SIZE
        self.assertComputeFails(TensorflowInputSiCNN(), [0.5] * 1)
        self.assertComputeFails(TensorflowInputSiCNN(), [0.5] * (FRAME_SIZE + 1))


suite = allTests(TestTensorflowInputSiCNN)

if __name__ == "__main__":
    TextTestRunner(verbosity=2).run(suite)
