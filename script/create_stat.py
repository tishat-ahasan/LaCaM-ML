import subprocess

# Run 'sudo apt update' and capture the output
map_name = "assets/warehouse.map"
agent = "40"
heuristic = "distance"
seed = 1
heuristics = ['ML','distance', 'conflict', 'neighbour']
file_path = 'Data/Supervised/32by32_updated.txt'
# file = open(file_path, 'a')

# header = 'obstacle_p,agent_p,a_g,a_ng,d_below,d_above,d_max,d_min,d_avg,d_std,c_below,c_above,c_max,c_min,c_avg,c_std,ng_0,ng_1,ng_2,ng_3,ng_4,y1,y2,y3'
# file.write(header+'\n')
total_nodes = 922.0
obstacles = 102.0

# print("here")
winners = {}
for agent in [50, 100, 250, 500, 1000]:
    # time = {'ML': 0,'distance': 0, 'conflict':0, 'neighbour':0}
    for seed in range(5):
        # print("Inside agent")
        Node = {}
        results = {}
        Y = {}
        for h in heuristics:
            Y[h] = 0
        for heuristic in heuristics:
            # print(seed, agent, heuristic)
            command = "build/main -m "+map_name+" -N "+ str(agent)+" -h "+heuristic+" -v 1 -s "+str(seed)
            try:
                result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
                # print(seed,",", heuristic,",", result, end="")
                # 'universal_newlines=True' converts the output to text
            except subprocess.CalledProcessError as e:
                result = e.output  # Capture the error message if 'sudo apt update' fails
                print("Error: ", result)

            # Print the output
            results[heuristic] = result
            stats = result.split(",")
            # Agent,Seed,Heuristic,Solved,HNode,LNode,Time,Makespan,SOC,SOL
            if "failed" in stats[0]:
                st = str(agent)+","+str(seed)+","+heuristic+",0"+",-1,-1,-1,-1,-1,-1"
                print(st)
            else:
                HNode_result = stats[0].split(":")[1]
                LNode_result = stats[1].split(":")[1]
                Time = stats[2].split(":")[1][:-2]
                Makespan = stats[3].split(":")[1]
                SOC = stats[4].split(":")[1]
                SOL = stats[5].split(":")[1][:-1]
                st = str(agent)+","+str(seed)+","+heuristic+",0"+","+HNode_result+","+LNode_result+","+Time+","
                print(f"{agent},{seed},{heuristic},{1},{HNode_result},{LNode_result},{Time},{Makespan},{SOC},{SOL}")
            # Time = stats[2].split(":")
            # Time = float(Time[1][:-2])
            # time[heuristic] += Time
            # Node[heuristic] = [HNode_result[1]]
            # Node[heuristic].append(LNode_result[1])
    #     # sorted_dict = dict(sorted(Node.items(), key=lambda item: (item[1][0], item[1][1])))
    #     # winner = next(iter(sorted_dict))
    #     winner, _ = min(Node.items(), key=lambda item: (item[1][0], item[1][1]))
    #     # print(seed, agent, Node, ",Winner: ", winner)
    #     winners[winner] = winners.get(winner, 0) + 1
    # print("For: ", agent, winners)
    # # print(time)
    # winners = {}


            
