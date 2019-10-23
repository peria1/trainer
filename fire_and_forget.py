# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:07:43 2019

@author: Bill
"""

import asyncio
import time

class status():
    def __init__(self):
        self.done = False

def fire_and_forget(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, *kwargs)

    return wrapped
#
# I do not understand why, but I can only get this to work using a decorator. I kind of
#    find decorators annoying. But I can't get the arguments to work correctly otherwise. 
#    At this point I guess I can just decorate the call to trainer, like below, and then 
#    I should be able to send commands to it from the Matplotlib interface. 
#    
#    Here what happens is 
#
@fire_and_forget
def foo(st):
    print('done is',st.done)
    while not st.done:
        time.sleep(1.0)
        print('looping...done is',st.done)
    print("foo() completed")
    return


stat = status()
print("Hello")
#fire_and_forget(foo,stat)
foo(stat)
#foo = fire_and_forget(foo)

print("I didn't wait for foo()")
