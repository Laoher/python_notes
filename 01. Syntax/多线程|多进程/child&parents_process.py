import os

print('Process (%s) starts' % os.getpid())

pid = os.fork()  # create a child process after calling fork(), child id will always be 0, and two processes will return 2 results
print(pid)
if pid == 0:
    print('I am child process (%s) and my parent is %s.' % (os.getpid(), os.getppid()))
else:
    print('I (%s) just created a child process (%s).' % (os.getpid(), pid))


#print('I am the child (%s) and my parents is (%s)' % (os.getpid()))