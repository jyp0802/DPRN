import numpy as np
import sys
import os

TOP_N = 200
DISP_N = 10

def all_worse(x, y):
    if x[2] > y[2]:
        return False
    for i in range(4):
        if x[3][i] > y[3][i]:
            return False
    return True

if len(sys.argv) != 2:
    print("Which file?")
    exit()

file_path = sys.argv[1]

res_dir = 'tune/breakfast/'+file_path

if not os.path.exists(res_dir):
    print("No such path!")
    exit()

wf = open('temp.txt', 'w')
wf.write(file_path+'\n')

bestdata = []

for idx, f_name in enumerate(sorted(os.listdir(res_dir))):
    bestdata.append([])

    f_path = os.path.join(res_dir, f_name)
    f = open(f_path)
    lines = f.readlines()

    SKIP = -1
    for i in range(len(lines)):
        if len(lines[i]) > 0 and lines[i][0] == '[':
            if SKIP == -1:
                SKIP = 0
            else:
                break
        if SKIP >= 0:
            SKIP += 1

    EPOCH = -1
    i = 0
    while i < len(lines):
        if EPOCH == -1:
            if len(lines[i]) > 0 and lines[i][0] == '[':
                EPOCH = 0
            else:
                i += 1
        if EPOCH >= 0:
            if len(lines[i]) > 0 and lines[i][0] == '[':
                EPOCH += 1
            else:
                break
            i += SKIP

    splitdata = []
    li = 0
    while True:
        if li >= len(lines):
            break

        while not 'TRN' in lines[li]:
            li += 1

        mname = lines[li].strip()

        modeldata = []

        vi = li
        # while vi < len(lines) and lines[vi+1].strip() != '':\
        for ep in range(EPOCH):
            vi += SKIP
            acc = float(lines[vi].split()[3][:-1])
            sdl = lines[vi].split()
            if len(sdl) == 17:
                cd = [float(sdl[3][:-1]), float(sdl[6][1:-1]), float(sdl[7][:-1]), float(sdl[8][:-2])]
                md = [float(sdl[11][:-1]), float(sdl[14][1:-1]), float(sdl[15][:-1]), float(sdl[16][:-1])]
            elif len(sdl) == 21:
                cd = [float(sdl[6][:-1]), float(sdl[9][:-1]), float(sdl[10][:-1]), float(sdl[11])]
                md = [float(sdl[15][:-1]), float(sdl[18][1:-1]), float(sdl[19][:-1]), float(sdl[20][:-1])]
            # elif len(sdl) == 30:
            #     cd = [float(sdl[6][:-1]), float(sdl[9][:-1]), float(sdl[10][:-1]), float(sdl[11])]
            #     md = [float(sdl[15][:-1]), float(sdl[18][1:-1]), float(sdl[19][:-1]), float(sdl[20][:-1])]
            else:
                print("line 89!")
                print(vi)
                print(len(sdl))
                print(sdl)
                exit()
            av = sum(cd) / len(cd)
            modeldata.append([mname, ep, acc, cd, md, av])

        # sdl = lines[vi].split()
        # maxd = [float(sdl[11][:-1]), float(sdl[14][1:-1]), float(sdl[15][:-1]), float(sdl[16][:-1])]

        # for mi, mv in enumerate(maxd):
        #     ei = EPOCH-1
        #     for ed in reversed(modeldata):
        #         if ed[4][mi] != mv:
        #             if not modeldata[ei+1] in bestdata[idx]:
        #                 bestdata[idx].append(modeldata[ei+1])
        #             break
        #         ei -= 1
        
        if len(splitdata) == 0:
            splitdata.append(modeldata[0])

        for xxx in modeldata[1:]:
            add = True
            for yyy in splitdata:
                if all_worse(xxx, yyy):
                    add = False
                    break
            if add:
                splitdata = [x for x in splitdata if not all_worse(x, xxx)]
                splitdata.append(xxx)

        li = vi+1

    bestdata[idx] = splitdata

    wf.write('\n')
    prev_model = ''
    for bd in bestdata[idx]:
        if bd[0] != prev_model:
            wf.write(bd[0]+'\n')
            prev_model = bd[0]
        wr_line = "epoch {} {:.4f} edit = {:.4f}, f1 = [{:.15f}, {:.15f}, {:.15f}]\n".format(bd[1], bd[2], *bd[3])
        wf.write(wr_line)

wf.close()

#####################################


data = [[]]
emdata = ['']

f = open('temp.txt', 'r')
lines = f.readlines()
i = 1
cur_model = 'NOT_INITIALIZED'
while i < len(lines):
    l = lines[i]
    if 'TRN' in l:
        split_data = []
        while i < len(lines) and len(lines[i]) > 1:
            if 'TRN' in lines[i]:
                cur_model = lines[i].strip()
                i += 1
                continue
            split_data.append([cur_model, lines[i]])
            i += 1
        # print(split_data)
        # raw_input()
        new_data = []
        new_emdata = []
        for sd in split_data:
            sdl = sd[1].split()
            d = [float(sdl[-1][:-1]), float(sdl[-2][:-1]), float(sdl[-3][1:-1]), float(sdl[2]), float(sdl[5][:-1])]
            for idx, od in enumerate(data):
                new_data.append(od + [d])
                # new_emdata.append(emdata[idx] + sd[0][MODELNAME_PREFIX:] + '-' + sdl[1] + ' ')
                new_emdata.append(emdata[idx] + sdl[1] + ' ')
        data = new_data
        emdata = new_emdata
    
    i += 1
    if i >= len(lines) or (len(lines[i]) > 1 and (not 'epoch' in lines[i][:5]) and (not 'TRN' in lines[i])):
        break

if len(data[0]) == 0:
    print('Add THISONE')
    exit()

mean_data = []
for d in data:
    d = np.array(d)
    mean_data.append(np.mean(d, 0).tolist())

for i in range(len(mean_data)):
    mean_data[i] = [mean_data[i], emdata[i]]

edit_sorted_data = sorted(mean_data, key=lambda x: x[0][-1], reverse=True)
edit_best_data = []
edit_best_data.append(edit_sorted_data[0])
for sb in edit_sorted_data[1:TOP_N*4]:
    skip = False
    for bb in edit_best_data:
        worse = True
        for i in range(len(sb[0])):
            if sb[0][i] > bb[0][i]:
                worse = False
                break
        if worse:
            skip = True
            break
    if skip:
        continue
    edit_best_data.append(sb)
    if len(edit_best_data) == TOP_N:
        break

f1_50_sorted_data = sorted(mean_data, key=lambda x: x[0][0], reverse=True)
f1_50_best_data = []
f1_50_best_data.append(f1_50_sorted_data[0])
for sb in f1_50_sorted_data[1:TOP_N*4]:
    skip = False
    for bb in f1_50_best_data:
        worse = True
        for i in range(len(sb[0])):
            if sb[0][i] > bb[0][i]:
                worse = False
                break
        if worse:
            skip = True
            break
    if skip:
        continue
    f1_50_best_data.append(sb)
    if len(f1_50_best_data) == TOP_N:
        break

f1_25_sorted_data = sorted(mean_data, key=lambda x: x[0][1], reverse=True)
f1_25_best_data = []
f1_25_best_data.append(f1_25_sorted_data[0])
for sb in f1_25_sorted_data[1:TOP_N*4]:
    skip = False
    for bb in f1_25_best_data:
        worse = True
        for i in range(len(sb[0])):
            if sb[0][i] > bb[0][i]:
                worse = False
                break
        if worse:
            skip = True
            break
    if skip:
        continue
    f1_25_best_data.append(sb)
    if len(f1_25_best_data) == TOP_N:
        break

f1_10_sorted_data = sorted(mean_data, key=lambda x: x[0][2], reverse=True)
f1_10_best_data = []
f1_10_best_data.append(f1_10_sorted_data[0])
for sb in f1_10_sorted_data[1:TOP_N*4]:
    skip = False
    for bb in f1_10_best_data:
        worse = True
        for i in range(len(sb[0])):
            if sb[0][i] > bb[0][i]:
                worse = False
                break
        if worse:
            skip = True
            break
    if skip:
        continue
    f1_10_best_data.append(sb)
    if len(f1_10_best_data) == TOP_N:
        break

acc_sorted_data = sorted(mean_data, key=lambda x: x[0][3], reverse=True)
acc_best_data = []
acc_best_data.append(acc_sorted_data[0])
for sb in acc_sorted_data[1:]:#[1:TOP_N*4]:
    skip = False
    for bb in acc_best_data:
        worse = True
        for i in range(len(sb[0])):
            if sb[0][i] > bb[0][i]:
                worse = False
                break
        if worse:
            skip = True
            break
    if skip:
        continue
    acc_best_data.append(sb)
    # if len(acc_best_data) == TOP_N:
    #     break

av_sorted_data = sorted(mean_data, key=lambda x: ((x[0][0]+x[0][1]+x[0][2]+x[0][3]*100+x[0][4])/5.0), reverse=True)
av_best_data = []
av_best_data.append(av_sorted_data[0])
for sb in av_sorted_data[1:TOP_N*4]:
    skip = False
    for bb in av_best_data:
        worse = True
        for i in range(len(sb[0])):
            if sb[0][i] > bb[0][i]:
                worse = False
                break
        if worse:
            skip = True
            break
    if skip:
        continue
    av_best_data.append(sb)
    if len(av_best_data) == TOP_N:
        break

cnt = 0
print("f1-10")
for md in f1_10_best_data:
    item_list = [md[0][2], md[0][1], md[0][0], md[0][4], md[0][3]*100, md[1]]
    print("{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{}".format(*item_list))
    cnt += 1
    if cnt == DISP_N:
        break
print('-')
print("f1-25")
cnt = 0
for md in f1_25_best_data:
    item_list = [md[0][2], md[0][1], md[0][0], md[0][4], md[0][3]*100, md[1]]
    print("{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{}".format(*item_list))
    cnt += 1
    if cnt == DISP_N:
        break
print('-')
print("f1-50")
cnt = 0
for md in f1_50_best_data:
    item_list = [md[0][2], md[0][1], md[0][0], md[0][4], md[0][3]*100, md[1]]
    print("{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{}".format(*item_list))
    cnt += 1
    if cnt == DISP_N:
        break
print('-')
print("edit")
cnt = 0
for md in edit_best_data:
    item_list = [md[0][2], md[0][1], md[0][0], md[0][4], md[0][3]*100, md[1]]
    print("{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{}".format(*item_list))
    cnt += 1
    if cnt == DISP_N:
        break
print('-')
print("acc1")
cnt = 0
newacclist = []
for md in acc_best_data:
    # item_list = [md[0][2], md[0][1], md[0][0], md[0][4], md[0][3]*100, md[1]]
    # print("{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{}".format(*item_list))
    # cnt += 1
    # if cnt == DISP_N:
    #     break
    item_list = [md[0][2], md[0][1], md[0][0], md[0][4], md[0][3]*100, md[1]]
    # if not (item_list[0] >= 69.7 and item_list[1] >= 66.5 and item_list[2] >= 56.0 and item_list[3] >= 67.2):
    # if not (item_list[4] >= 86.75 and item_list[4] < 86.85):
    #     continue
    # if not (item_list[0] >= 87.15 and item_list[4] < 87.25):
    #     continue
    # if not (item_list[1] >= 85.75 and item_list[4] < 85.85):
    #     continue
    # if not (item_list[2] >= 79.05 and item_list[4] < 79.15):
    #     continue
    # if not (item_list[3] >= 80.75 and item_list[4] < 80.85):
    #     continue
    # if not (item_list[0] >= 87.15):
    #     continue
    # if not (item_list[1] >= 85.75):
    #     continue
    # if not (item_list[2] >= 78.8):
    #     continue
    # if not (item_list[3] >= 80.):
    #     continue
    if not (item_list[4] >= 71.35):
        continue
    item_list = [round(md[0][2]*10), round(md[0][1]*10), round(md[0][0]*10), round(md[0][4]*10), round(md[0][3]*100*10)] + item_list
    newacclist.append(item_list)
    # print("{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{}".format(*item_list))
    # cnt += 1
    # if cnt == DISP_N*3:
    #     break
newacclist = sorted(newacclist, key=lambda x: sum(x[:5]), reverse=True)
# newacclist = sorted(newacclist, key=lambda x: x[2], reverse=True)


cnt = 0
for md in acc_best_data:
    item_list = [md[0][2], md[0][1], md[0][0], md[0][4], md[0][3]*100, md[1]]
    print("{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{}".format(*item_list))
    cnt += 1
    if cnt == DISP_N:
        break

print('-')
print("acc2")
cnt = 0
for item_list in newacclist:
    print("{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{}".format(*item_list[5:]))
    cnt += 1
    if cnt == DISP_N:
        break

print('-')
print("av")
cnt = 0
for md in av_best_data:
    item_list = [md[0][2], md[0][1], md[0][0], md[0][4], md[0][3]*100, md[1]]
    # if not (item_list[0] >= 69.7 and item_list[1] >= 66.5 and item_list[2] >= 56.0 and item_list[3] >= 67.2):
    #     continue
    print("{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{}".format(*item_list))
    cnt += 1
    if cnt == DISP_N:
        break