import sys

def is_number(n):
    try:
        float(n)
        return True
    except ValueError:
        return False

total = 0
count = 0

for line in sys.stdin:
    if is_number(line):
        total += float(line)
        count += 1
    else:
        if total > 0 and count > 0:
            print total/count
        total = 0
        count = 0
        if len(line) > 3:
            print line
        sys.stdout.flush()

