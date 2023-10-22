import subprocess

# Run 'sudo apt update' and capture the output
map_name = "assets/random-32-32-10.map"
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
for agent in [50, 100, 150, 200, 250, 300]:
    for seed in range(100):
        # print("Inside agent")
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
                print("Error: ", result)

            # Print the output
            results[heuristic] = result
            stats = result.split(",")
            HNode_result = stats[0].split(":")
            LNode_result = stats[1].split(":")
            Node[heuristic] = [HNode_result[1]]
            Node[heuristic].append(LNode_result[1])
        # sorted_dict = dict(sorted(Node.items(), key=lambda item: (item[1][0], item[1][1])))
        # winner = next(iter(sorted_dict))
        winner, _ = min(Node.items(), key=lambda item: (item[1][0], item[1][1]))
        # print(seed, agent, Node, ",Winner: ", winner)
        winners[winner] = winners.get(winner, 0) + 1
    print("For: ", agent, winners)
    winners = {}


            
