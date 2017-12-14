# KDDCup 2015 experiments dataset


## Tensorflow

The models are tested with tensorflow 1.4


## To run test

You need the library nosetests. Test are oriented to check the model does not fail and
keeps certain properties. However, they do not check the correctness of the
implementation.

Examples:

```
KDDCup$ nosetests tests/
KDDCup$ nosetests tests/test_seq_lstm.py
KDDCup$ nosetests --nologcapture tests.test_seq_lstm.SeqLSTMModelTest.test_fit_loss
```