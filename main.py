# !/usr/bin/env python
# -*- coding: utf-8 -*-
# 2022/6/16 9:17
import silence_tensorflow.auto

from tools.parser import parseArgs
from tools.processor_ensemble import processModel as processor_ensemble

if __name__ == '__main__':
    args = parseArgs().args

    process = processor_ensemble(args)

    process.start()