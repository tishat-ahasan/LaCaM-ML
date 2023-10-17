import subprocess

# Run 'sudo apt update' and capture the output
map_name = "assets/random-32-32-10.map"
agent = "40"
heuristic = "distance"
seed = 1
heuristics = ['distance', 'conflict', 'neighbour']
file_path = 'Data/Supervised/32by32.txt'
file = open(file_path, 'a')

header = 'obstacle_p,agent_p,a_g,a_ng,d_max,d_min,d_avg,d_std,c_max,c_min,c_avg,c_std,ng_0,ng_1,ng_2,ng_3,ng_4,y1,y2,y3'
file.write(header+'\n')
total_nodes = 922.0
obstacles = 102.0

for seed in range(100):
    for agent in [30, 40, 50, 60, 70, 80, 90, 100, 120, 140]:
        Node = {}
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
                # 'universal_newlines=True' converts the output to text
            except subprocess.CalledProcessError as e:
                result = e.output  # Capture the error message if 'sudo apt update' fails

            # Print the output
            r1 = result.split('\n')
            results[heuristic] = r1[:-2]
            stats = r1[-2].split(",")
            HNode_result = stats[0].split(":")
            LNode_result = stats[1].split(":")
            Node[heuristic] = [HNode_result[1]]
            Node[heuristic].append(LNode_result[1])
        sorted_dict = dict(sorted(Node.items(), key=lambda item: (item[1][0], item[1][1])))
        print(seed, agent, sorted_dict)
        winner = next(iter(sorted_dict))
        # print("Winner: ", winner)
        Y[winner] = 1
        label = ""
        for k,v in Y.items():
            label +=","+str(v)
        header = str(obstacles/total_nodes)+","+str(agent/total_nodes)+","
        outlier = int(len(results[winner])*0.1)
        for  i in range(len(results[winner])):
            if i > outlier and i+outlier<len(results[winner]):
                stat = results[winner][i]
                line = header+stat+label
                file.write(line + '\n')
file.close()


            
