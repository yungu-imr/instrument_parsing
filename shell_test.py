import os

for i in ['binary', 'parts']:
    for j in [0, 1, 2, 3]:
        cmd = 'python main.py --problem_type {} --fold {} --model DeepLabV3'.format(i, j)
        print(cmd)
        os.system(cmd)
