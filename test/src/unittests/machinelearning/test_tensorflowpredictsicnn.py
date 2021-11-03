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

SAMPLE_RATE = 22050


class TestTensorFlowPredictSiCNN(TestCase):

    def testRegressionFrozenModel(self):
        expected = numpy.load(join(filedir(), 'tensorflowpredictsicnn', 'dubstep_expected.npy'))

        filename = join(testdata.audio_dir, 'recorded', 'dubstep.wav')
        model = join(testdata.models_dir, 'sicnn', 'VGG91_a125bis.pb')

        audio = MonoLoader(filename=filename, sampleRate=SAMPLE_RATE)()
        found = TensorflowPredictSiCNN(graphFilename=model, patchHopSize=0)(audio)
        found = numpy.mean(found, axis=0)

        # Relatively high tolerance due to the deviations in the reproduced mel-spectrograms.
        # We need further QA to understand it this difference is enough to change the sense of
        #  the predictions in corner cases.
        self.assertAlmostEqualVector(found, expected, 1e-1)

    def testEmptyModelName(self):
        # With empty model names the algorithm should skip the configuration without errors.
        self.assertConfigureSuccess(TensorflowPredictSiCNN(), {})
        self.assertConfigureSuccess(TensorflowPredictSiCNN(), {'graphFilename': ''})
        self.assertConfigureSuccess(TensorflowPredictSiCNN(), {'graphFilename': '',
                                                               'input': '',
                                                               })
        self.assertConfigureSuccess(TensorflowPredictSiCNN(), {'graphFilename': '',
                                                               'input': 'wrong_input'
                                                               })
        self.assertConfigureSuccess(TensorflowPredictSiCNN(), {'savedModel': ''})
        self.assertConfigureSuccess(TensorflowPredictSiCNN(), {'savedModel': '',
                                                               'input': '',
                                                               })
        self.assertConfigureSuccess(TensorflowPredictSiCNN(), {'savedModel': '',
                                                               'input': 'wrong_input',
                                                               })
        self.assertConfigureSuccess(TensorflowPredictSiCNN(), {'graphFilename': '',
                                                               'savedModel': '',
                                                               })
        self.assertConfigureSuccess(TensorflowPredictSiCNN(), {'graphFilename': '',
                                                               'savedModel': '',
                                                               'input': '',
                                                               })
        self.assertConfigureSuccess(TensorflowPredictSiCNN(), {'graphFilename': '',
                                                               'savedModel': '',
                                                               'input': 'wrong_input',
                                                               })


    def testInvalidParam(self):
        model = join(testdata.models_dir, 'vgg', 'vgg4.pb')
        self.assertConfigureFails(TensorflowPredictSiCNN(), {'graphFilename': model,
                                                             'input': 'wrong_input_name',
                                                             'output': 'model/Softmax',
                                                             })  # input do not exist in the model
        self.assertConfigureFails(TensorflowPredictSiCNN(), {'graphFilename': 'wrong_model_name',
                                                             'input': 'model/Placeholder',
                                                             'output': 'model/Softmax',
                                                             })  # the model does not exist

suite = allTests(TestTensorFlowPredictSiCNN)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
