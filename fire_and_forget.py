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
# I do not understand why, but I can only get this to work using a decorator. 
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
print("I didn't wait for foo()")
