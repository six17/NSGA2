import math
import numpy as np
from gantt import Gantt
import json

class Platform:                     # equal to resource
    def __init__(self, platNum, speed, initPos):
        #self.ptype = ptype          # platform type
        self.platNum = platNum      # platform number, from 0 to m
        self.speed = speed          # (average) moving speed, m/s
        self.initPos = initPos      # platform initial position, tuple
    
    def exec_order(self, Tasks, priority):
        # To calculate the execution order for each platform
        # Tasks: all the Tasks in a list
        # priority: the priority-relation among all the Tasks in a list
        Tasks_prior = []
        for prior_ind in priority:
            Tasks_prior.append(Tasks[prior_ind])
        self.execOder = []          # order of task execution (task number) in a list
        self.execPos = []           # execution position, tuple in a list
        for Task in Tasks_prior:
            if Task.resInv[self.platNum] > 0:
                self.execOder.append(Tasks.index(Task))
                self.execPos.append(Task.pos)


class Task0:
    def __init__(self, taskNum, pred, pos, resInv):
        self.taskNum = taskNum      # task number, from 0 to n
        self.pred = pred            # indices of direct predecessors, None or list
        #self.dur = dur              # duration, scaler (calc. by Task.duration(...))
        self.pos = pos              # execution position, tuple
        self.resInv = resInv        # resource/platform invested, nparray, by NSGA
    
    def duration(self, Platforms):
        # To calculte the duration of each Task (first by ED, later by situAware)
        # Platforms: all the platforms in number order in a list
        platTime = []               # time to consume for each platform
        resIdx = np.nonzero(self.resInv)[0]
        for idx in resIdx:
            Platform = Platforms[idx]
            if Platform.execOder.index(self.taskNum) == 0:      # initial position
                platPos = Platform.initPos
            else:
                platPos = Platform.execPos[Platform.execOder.index(self.taskNum)-1] # last task position
            dist = math.sqrt((self.pos[0]-platPos[0])**2+(self.pos[1]-platPos[1])**2)
            time = dist/Platform.speed
            platTime.append(time)
        if platTime==[]:
            self.dur= 9999
        else:
            self.dur = max(platTime)

    def start_time(self, T_pred, T_prior, resAvail):
        # To schedule Task at the earliest feasible time
        # T_pred is the Tasks of the direct predecessors in a list
        # T_prior is all the prior Tasks in a list, by NSGA
        # resAvail is all the resource available in a list/nparray 
        if T_prior is None:
            self.ts = 0                     # Task start time
        else:
            if T_pred is None:
                te_pred_max = 0
            else:
                te_pred = []                # end time of the direct predecessors
                for Task in T_pred:
                    te_pred.append(Task.ts+Task.dur)
                te_pred_max = max(te_pred)  # from te_pred_max on ...
            tste_prior = []
            for Task in T_prior:
                if Task.ts >= te_pred_max:
                    tste_prior.append(Task.ts)
                if Task.ts+Task.dur >= te_pred_max:
                    tste_prior.append(Task.ts+Task.dur)
            tste_prior.sort()
            for beginTime in tste_prior:
                resCsm1 = np.zeros_like(self.resInv) # resource consumed
                for Task in T_prior:
                    if beginTime >= Task.ts and beginTime < Task.ts+Task.dur:               # begin check
                        resCsm1 += Task.resInv
                resCsm1 += self.resInv
                cmpRst1 = (resCsm1 <= resAvail)
                if cmpRst1.all():
                    endTime = beginTime + self.dur
                    errorFlag = 0
                    for middleTime in tste_prior:
                        if middleTime > beginTime and middleTime < endTime:
                            resCsm2 = np.zeros_like(self.resInv)
                            for Task in T_prior:
                                if middleTime >= Task.ts and middleTime < Task.ts+Task.dur: # middle check
                                    resCsm2 += Task.resInv
                            resCsm2 += self.resInv
                            cmpRst2 = (resCsm2 <= resAvail)
                            if cmpRst2.all():
                                continue
                            else:
                                errorFlag = 1
                                break
                    if errorFlag == 0:
                        self.ts = beginTime
                        break


def schedule(Tasks, priority):
    # To calculate the feasible start time for each Task
    # Tasks: all the Tasks in a list
    # priority: the priority-relation among all the Tasks in a list
    Tasks_prior = []
    for prior_ind in priority:
        Tasks_prior.append(Tasks[prior_ind])
    for (ind, Task) in enumerate(Tasks_prior):
        if Task.pred is None:
            T_pred = None
        else:
            T_pred = []
            for i in Task.pred:
                T_pred.append(Tasks[i])
        if ind == 0:
            T_prior = None
        else:
            T_prior = Tasks_prior[:ind]
        Task.start_time(T_pred, T_prior, resAvail)


def calc_makespan(Tasks):
    # To calculate the Makespan
    # Tasks: all the Tasks in a list
    te = []             # end time of all the tasks
    for Task in Tasks:
        te.append(Task.ts+Task.dur)
    return max(te)


def calc_winrate(Tasks, Situation):
    ...


def plot_gantt(Tasks):
    # To draw the Gantt chart of the COA (gantt)
    # Tasks: all the Tasks in a list
    pkg_kvs = []        # key-value in 'packages'
    for Ti, Task in enumerate(Tasks):
        resIdx = np.nonzero(Task.resInv)[0] # platforms invested
        for idx in resIdx:
            kv = {}
            kv['label'] = 'P' + str(idx)    # still have a problem!!!
            kv['task'] = 'T' + str(Ti)
            kv['start'] = Task.ts
            kv['end'] = Task.ts + Task.dur
            pkg_kvs.append(kv)
    data = {'packages': pkg_kvs, 'title': 'Course of Action',
                'xlabel': 'Time'}
    jsData = json.dumps(data, indent=4, separators=(',', ': '))
    filename = 'GanttData.json'
    fileobj = open(filename, 'w')
    fileobj.write(jsData)
    fileobj.close()
    gt = Gantt('./'+filename)
    gt.render()
    gt.show()

# rd=np.random.RandomState(0) 
# resAvail=rd.randint(3,4,size=1)
# R0=resAvail
# resAvail=[1]*resAvail[0]
# print(resAvail)
# tasknum=rd.randint(4,5,size=1)
# pos=rd.randint(0,51,(tasknum[0],2))
# print(pos)
# mapmatrix=rd.randint(0,2,(tasknum[0],tasknum[0]))
# mapmatrix=np.triu(mapmatrix,1)
# print(mapmatrix)
# v=rd.random((1,R0[0]))*5+0.5
# print(v)
# workmap=list()

# for i in range (0,tasknum[0]):
#     a=np.where(mapmatrix[:,i]==1)

#     b=list()
#     for i in range (0,np.size(a)):
#         b.append(a[0][i]+1)
#     workmap.append(b)   
# workmap=np.array([x for x in workmap])
# print(workmap)



# #任务生成
# def workrank(vars,workmap):
#     size=np.size(vars)
#     size1=size
#     rank=np.array([0]*size)
#     for i in range(0,size):
#         for j in range(0,size1):
#             flag=0 
#             for k in range(0,np.size(workmap[vars[j]-1])):
#                 if workmap[vars[j]-1][k] not in rank:
#                     flag=1     
#                     break
#             if flag == 0:
#                 rank[i]=vars[j]
#                 vars=np.delete(vars,j)
#                 size1=size1-1
#                 break               
#     return rank     
# Vars = np.array([[1,1,1,4,2,3,1,0,0,1,1,1,0,0,1,1,0,0]]) # 得到决策变量矩阵
# g=2

# for i in range (0,Vars.shape[0]):
#     rank=workrank(Vars[i,g:np.size(workmap)+g].astype('int64'),workmap)
#     priority=rank-1
#     r=Vars[i,np.size(workmap)+g:np.size(workmap)+ np.size(workmap) * R0[0] +g].reshape(np.size(workmap),R0[0])

#     rs=8/(np.sum(r)+1)
#     resInv=[]
#     for j in range(0,tasknum[0]):
        
#         resInv.append(np.array([x for x in r[j,:]]))   
#     Tasks=[]
#     for k in range (0,tasknum[0]):
#         if workmap[k]==[]:
#             Tasks.append(Task0(k,None,pos[k,:],resInv[k]))
#         else:
#             Tasks.append(Task0(k,[x-1 for x in workmap[k]],pos[k,:],resInv[k]))
#     Plaforms=[]
#     for m in range(0,R0[0]):
#         P = Platform(m, v[0][m], (0,0))
#         P.exec_order(Tasks, priority)
#         Plaforms.append(P)

#     for Task in Tasks:
#         Task.duration(Plaforms)
#     schedule(Tasks, priority)     
#     plot_gantt(Tasks)

resAvail = [1, 1, 1]
priority = [0,1,2,3]         # indices start from 0, in line with np
pred0 = None
pred1 = None
pred2 = None
pred3 = [0,2]
pos0 = (44, 47)
pos1 = (0, 3)
pos2 = (3,39)
pos3 = (9, 19)
resInv0 = np.array([1., 0., 0.])
resInv1 = np.array([1., 1., 1.])
resInv2 = np.array([0., 0., 1.])
resInv3 = np.array([1., 0., 0.])
T0 = Task0(0, pred0, pos0, resInv0)
T1 = Task0(1, pred1, pos1, resInv1)
T2 = Task0(2, pred2, pos2, resInv2)
T3 = Task0(3, pred3, pos3, resInv3)
Tasks = [T0, T1, T2, T3]        # should be [T0, T1, ..., Tn-1] in order

v0 = 5.12798319
v1 = 0.85518029
v2 = 0.9356465
iPos0 = (0, 0) 
iPos1 = (0, 0) 
iPos2 = (0, 0) 
P0 = Platform(0, v0, iPos0)
P0.exec_order(Tasks, priority)
P1 = Platform(1, v1, iPos1)
P1.exec_order(Tasks, priority)
P2 = Platform(2, v2, iPos2)
P2.exec_order(Tasks, priority)
Plaforms = [P0, P1, P2]

for Task in Tasks:
    Task.duration(Plaforms)

schedule(Tasks, priority)
for i in range (0,4):
    print(Tasks[i].ts)
print(calc_makespan(Tasks))
plot_gantt(Tasks)