#!/usr/bin/env python3
import sys
import re

if len(sys.argv) != 3:
    print("Error: takes two arguments: path to the javascript file for which to toggle assert calls on/off and output file for results")
    sys.exit(1)
with open(sys.argv[1]) as f:
    lines = f.readlines()
# From the beginning of the line, after 0 or more whitespace characters, the first 'assert' string
assert_pattern_on = re.compile(r'^[ \t]*?(assert).*')
assert_pattern_off = re.compile(r'^[ \t]*?(' + re.escape('/* ') + ')assert')
comment_end_pattern = re.compile(re.escape(' */') + '$')
with open(sys.argv[2], 'w') as f:
    for line in lines:
        assert_expr = re.search(assert_pattern_on, line)
        comment = re.search(assert_pattern_off, line)
        if assert_expr:
            # Add comment
            line = (
                # Indentation
                line[:assert_expr.start(1)]
                # Start comment
                + '/* '
                # Assertion call up to newline
                + line[assert_expr.start(1):assert_expr.end(0)]
                # End comment
                + ' */'
                # Newline
                + line[assert_expr.end(0):]
            )
        elif comment:
            # Remove comment
            comment_end = re.search(comment_end_pattern, line)
            line = line[:comment.start(1)] + line[comment.end(1):comment_end.start(0)] + line[comment_end.end(0):]
        print(line, end='', file=f)
