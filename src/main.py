# -*- coding: utf-8 -*-


import os, sys

def main():
    os.environ['DEMENTIA_CONFIG_FILE'] = sys.argv[1] if len(sys.argv) > 1 else 'configure.json'
    import web
    
    web.webStart()

def test():
    os.environ['DEMENTIA_CONFIG_FILE'] = sys.argv[1] if len(sys.argv) > 1 else 'configure.json'
    import diagnosis
    
    diagnosis.execute(['fa352139-508c-40bf-a49d-db57de4f4a0b_0.wav',
                       'fa352139-508c-40bf-a49d-db57de4f4a0b_1.wav',
                       'fa352139-508c-40bf-a49d-db57de4f4a0b_2.wav'])

    pass
    
if __name__ == '__main__':
    
    #main()
    test()
