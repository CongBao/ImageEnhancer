""" Check and revise user inputs """

import re

__author__ = 'Cong Bao'

class Correction(object):
    """ The class used to correct user inputs """

    def __init__(self):
        self.back_slash = re.compile(r'\\+', re.IGNORECASE)
        self.end_slash = re.compile(r'^.*[^/]$', re.IGNORECASE)

    def replace_backslash(self, line):
        """ replace backslashes to slashes
            :param line: the input line
            :return: string after correction
        """
        return self.back_slash.sub('/', line)

    def add_endslash(self, line):
        """ add a slash in the end of line
            :param line: the input line
            :return: string after correction
        """
        if self.end_slash.match(line):
            return line + '/'
        return line

    def correct(self, line):
        """ do all corrections
            :param line: the input line
            :return: string after correction
        """
        line = self.replace_backslash(line)
        line = self.add_endslash(line)
        return line
