import sys
import numpy as np
import agent_journal

argue = True

if argue == True:
    students = [sys.argv[1]]
    class_periods = [sys.argv[2]]
    studentlabels = str(int(sys.argv[1]))
    savename = "run_results/N=" + str(studentlabels) + "_per=" + str(sys.argv[2]) +".npy"
    
else:
    students = [33.5*(10**3), 2000, 65*(10**3)]
    studentlabels = ["medium (Ft. Collins)", "small", "very large"]
    class_periods = [i for i in range(3,6)]
    savename = "test_C3.npy"

class_sizes = [i*20 + 10 for i in range(5)]
majors = [False,  100, 500]

contacts = np.zeros((len(students), len(class_periods), len(class_sizes), len(majors)))
for N in range(len(students)):
    for pers in range(len(class_periods)):
        for szs in range(len(class_sizes)):
            for maj in range(len(majors)):
                
                print(students[N])
                print("repeat")

                model = agent_journal.UnivModel(int(students[N]), 5, int(np.floor(int(students[N]) / 2)),
                                                class_periods[pers], class_sizes[szs], majors[maj])
                for i in range(pers):
                    model.step()
                    
                C = agent_journal.contactnumbers(model, returnarr = False)[-1]
                contacts[N, pers, szs, maj] = C

np.save(savename, C)

