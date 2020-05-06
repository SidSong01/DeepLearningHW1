# this is the tensorflow implementation of seq2seq_v2t.

## Version

* Python 3.6.0
* Tensorflow 1.6.0



## Required Packages

You need to pip3 install these packages below to run this code.

```python
import tensorflow as tf # keras.preprocessing included
import numpy as np
import pandas as pd
import argparse
import pickle
from colors import *
from tqdm import *
```

# the training data: https://drive.google.com/open?id=1sSFbOU928jYp1xGx4PF4_hV8_w2kDQ-j

how to run the .sh:

Hw2_seq2seq.sh _ _ _
(•	_ _ _ here represent "the data directory", "the test data directory", "and the name of the output(.txt)"
the name of the output(.txt) = final_output.txt)


please read my .sh file


# reference: http://www.cs.utexas.edu/users/ml/papers/venugopalan.iccv15.pdf

the bleu score is about 0.7

detailed description please refer to the pdf report, thanks
