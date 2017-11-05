# -*- coding: utf-8 -*-  

from __future__ import print_function

"""
Created by Wang Han on 2017/11/5 15:46.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""
from __future__ import absolute_import

from utils import Logger

logger = Logger(filename='/home/wanghan/home/wanghan/Workspace/DeepBC/result.log').get_logger()

print(logger)

logger.info('hello')
logger.debug('hhh')
