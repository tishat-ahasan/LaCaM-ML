#include "../include/planner.hpp"
#include <cmath>
#include<algorithm>

#include<torch/torch.h>


uint Constraint::LNODE_CNT = 0;

Constraint::Constraint() : who(std::vector<int>()), where(Vertices()), depth(0)
{
}

Constraint::Constraint(Constraint* parent, int i, Vertex* v)
    : who(parent->who), where(parent->where), depth(parent->depth + 1)
    
{
  ++LNODE_CNT;
  who.push_back(i);
  where.push_back(v);
}

Constraint::~Constraint(){};

uint Node::HNODE_CNT = 0;


torch::jit::script::Module Node::net = torch::jit::load("./Data/models/traced_model.pt");
Node::Node(Config _C, DistTable& D, const std::string& _h, Node* _parent)
    : C(_C),
      parent(_parent),
      priorities(C.size(), 0),
      order(C.size(), 0),
      search_tree(std::queue<Constraint*>()),
      h(_h)
      {
  if (parent == nullptr){
    depth = 1;
  }
  else{
    parent->child++;
    depth = parent->depth+1;
    // std::cout<<"Child: "<<parent->child<<"\n";
  }
  
  ++HNODE_CNT;
  search_tree.push(new Constraint());
  const auto N = C.size();

  bool is_print = false;
  bool generate_dataset = false;

  // agents at goal or non goal
  float a_g = N;
  float a_ng = 0;
  std::vector<double> man_dist(N);

  for (size_t i = 0; i < N; ++i) {
    float dist = D.get(i, C[i]);
    if ( dist!= 0) {  
      man_dist[i] = dist;
      a_ng++;
    }
  }
  a_g -= a_ng;
  a_g /= N;
  a_ng /= N;

  

  // Manhattan Distance
  double below_avg_dist = 0;
  
  double max_dist = *std::max_element(man_dist.begin(), man_dist.end());
  double min_dist = *std::min_element(man_dist.begin(), man_dist.end());
  double average_dist = std::accumulate(man_dist.begin(), man_dist.end(), 0.0) / N;
  double sumOfSquares = 0.0;
  for (const double value : man_dist) {
      double diff = value - average_dist;
      sumOfSquares += diff * diff;
      if (diff<0)
        below_avg_dist++;
  }
  double std_dist = std::sqrt(sumOfSquares / N);



  // conflicts


  double max_conf = *std::max_element(D.conf_count.begin(), D.conf_count.end());
  double min_conf = *std::min_element(D.conf_count.begin(), D.conf_count.end());
  double average_conf = std::accumulate(D.conf_count.begin(), D.conf_count.end(), 0.0) / N;
  sumOfSquares = 0.0;
  double below_avg_conf = 0 ;
  for (const double value : D.conf_count) {
      double diff = value - average_conf;
      sumOfSquares += diff * diff;
      if (diff<0) {
        below_avg_conf++;
      }
  }
  double std_conf = std::sqrt(sumOfSquares / N);

  // neighbour
  std::vector<float> neighbour(5, 0.0);
  for (size_t i = 0; i < N; ++i) {
    neighbour[Planner::option[i]]++;
  }

  
  // 'obstacle_p', 'agent_p', 'a_g', 'a_ng', 
  // below_avg_dist, above_avg_dist, 'd_max_agent','d_min_agent', 'd_avg_agent', 'd_std_agent', 
  // below_avg,conf, above_avg_conf, 'c_max_agent','c_min_agent', 'c_avg_agent', 'c_std_agent', 
  // 'd_max_node', 'd_min_node','d_avg_node', 'd_std_node', 
  // 'c_max_node', 'c_min_node', 'c_avg_node','c_std_node', 
  // 'ng_0', 'ng_1', 'ng_2', 'ng_3', 'ng_4',

  // // Print Stats....
  // if (HNODE_CNT%5 == 1){
  //   std::cout<<a_g<<","<<a_ng<<",";
  //   std::cout<<below_avg_dist/N<<","<<1-(below_avg_dist/N)<<","<<max_dist<<","<<min_dist<<","<<average_dist<<","<<std_dist<<",";
  //   std::cout<<below_avg_conf/N<<","<<1-(below_avg_conf/N)<<","<<max_conf<<","<<min_conf<<","<<average_conf<<","<<std_conf<<",";
  //   for (int i = 0; i<4; ++i){
  //     std::cout<<neighbour[i]<<",";
  //   }
  //   std::cout<<neighbour[4]<<"\n";}

  // ************************************************ Distance ******************************************
  if (h == "distance"){
    if (is_print) std::cout<<h<<"\n";
    if (parent == nullptr) {
      // initialize
      for (size_t i = 0; i < N; ++i) priorities[i] = (float)D.get(i, C[i]) / N;
    } else {
      // dynamic priorities, akin to PIBT
      for (size_t i = 0; i < N; ++i) {
        if (D.get(i, C[i]) != 0) {
          priorities[i] = parent->priorities[i] + 1;
        } else {
          priorities[i] = parent->priorities[i] - (int)parent->priorities[i];
        }
      }
    }

    // set order
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
                [&](int i, int j) { return priorities[i] > priorities[j]; });}

  // ************************************************ Random ******************************************

  else if (h == "random"){
      if (is_print) std::cout<<h<<"\n";
    //random
      if (parent == nullptr) {
        // initialize
        for (size_t i = 0; i < N; ++i) priorities[i] = (float)D.get(i, C[i]) / N;
      } else {
        // dynamic priorities, akin to PIBT
        for (size_t i = 0; i < N; ++i) {
          if (D.get(i, C[i]) != 0) {
            priorities[i] = std::mt19937(std::random_device()())() % N + 1;
          } else {
            priorities[i] = parent->priorities[i] - (int)parent->priorities[i];
          }
        }
      }
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(),
                [&](int i, int j) { return priorities[i] > priorities[j]; });


      // set order based upon distance from goal
      for (uint i = 0; i < N; ++i) priorities[i] = (float)D.get(i, C[i]) / N;
      // // set order
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(),
                [&](int i, int j) { return priorities[i] > priorities[j]; });}

  // ************************************************ Conflict ******************************************

  else if (h=="conflict"){
      if (is_print) std::cout<<h<<"\n";
    //set order using conflict
      if (parent == nullptr) {
        // initialize
        for (uint i = 0; i < N; ++i) priorities[i] = (float)D.get(i, C[i]) / N;
      } else {
        // dynamic priorities, akin to PIBT
        for (size_t i = 0; i < N; ++i) {
          if (D.get(i, C[i]) != 0) {
            priorities[i] = D.get_conf(i);
          } else {
            priorities[i] = parent->priorities[i] - (int)parent->priorities[i];
          }
        }
      }
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(),
                [&](int i, int j) { return priorities[i] > priorities[j]; });}
  // ************************************************ Neighbour ******************************************

  else if (h == "neighbour"){
      if (is_print) std::cout<<h<<"\n";
      if (parent == nullptr) {
        // initialize
        for (uint i = 0; i < N; ++i) priorities[i] = (float)D.get(i, C[i]) / N;
      } else {
        // dynamic priorities, akin to PIBT
        for (size_t i = 0; i < N; ++i) {
          if (D.get(i, C[i]) != 0) {
            priorities[i] = 5 - Planner::option[i];
          } else {
            priorities[i] = parent->priorities[i] - (int)parent->priorities[i];
          }
        }
      }
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(),
                [&](int i, int j) { return priorities[i] > priorities[j]; });}

  else if (h == "ML"){

    //    'obstacle_p', 'agent_p', 'a_g', 'a_ng', 'd_below', 'd_above', 1-6
    //    'd_max_agent', 'd_min_agent', 'd_avg_agent', 'd_std_agent', 'c_below', 7-11
    //    'c_above', 'c_max_agent', 'c_min_agent', 'c_avg_agent', 'c_std_agent', 12-16
    //    'd_max_node', 'd_min_node', 'd_avg_node', 'd_std_node', 'c_max_node', 17-21
    //    'c_min_node', 'c_avg_node', 'c_std_node', 'ng_0', 'ng_1', 'ng_2', 22-27
    //    'ng_3', 'ng_4' 28-29
    // total 29

    
    // determine alpha, beta, gamma
    if (parent != nullptr and parent->child == 1){
      float total_nodes = 922.0;
      float obstacles = 102.0;
      torch::Tensor x = torch::tensor({
        float(obstacles/total_nodes),            //1
        float(N/total_nodes),                    //2
        float(a_g),                              //3
        float(a_ng),                             //4
        float(below_avg_dist/N),                 //5
        float(1-(below_avg_dist/N)),             //6
        float(max_dist/N),                       //7
        float(min_dist/N),                       //8
        float(average_dist/N),                   //9
        float(std_dist/N),                       //10
        float(below_avg_conf/N),                 //11
        float(1-(below_avg_conf/N)),             //12
        float(max_conf/N),                       //13
        float(min_conf/N),                       //14
        float(average_conf/N),                   //15
        float(std_conf/N),                       //16
        float(max_dist/total_nodes),             //17
        float(min_dist/total_nodes),             //18
        float(average_dist/total_nodes),         //19
        float(std_dist/total_nodes),             //20
        float(max_conf/total_nodes),             //21
        float(min_conf/total_nodes),             //22
        float(average_conf/total_nodes),         //23
        float(std_conf/total_nodes),             //24
        float(neighbour[0]/N),                   //25
        float(neighbour[1]/N),                   //26
        float(neighbour[2]/N),                   //27
        float(neighbour[3]/N),                   //28
        float(neighbour[4]/N)                    //29
      });
      x = x.unsqueeze(0);
      std::vector<torch::jit::IValue> input; input.push_back(x);
      torch::Tensor out = net.forward(input).toTensor();
      parent->alpha = out[0][0].item<float>();
      parent->beta = out[0][1].item<float>();
      parent->gamma = out[0][2].item<float>();
      if (parent->alpha < parent->beta && parent->alpha < parent->gamma) parent->alpha = 0;
      if (parent->beta < parent->alpha && parent->beta < parent->gamma) parent->beta = 0;
      if (parent->gamma < parent->alpha && parent->gamma < parent->beta) parent->gamma = 0; 
    }

      double lower_bound = 0.0001;
      double upper_bound = 0.0009;
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<double> dist(lower_bound, upper_bound);

    // std::cout<<"For Node: "<<HNODE_CNT<<"\n";
    // std::cout<<"("<<alpha<<", "<<beta<<", "<<gamma<<")"<<"\n";
    std::vector<float> CP(N);

    if (parent == nullptr){
      for (size_t i = 0; i < N; ++i) {
        priorities[i] = (float)D.get(i, C[i]) / N;
        CP[i] = priorities[i];
      }
    } 
    else {
      double max_init = *std::max_element(parent->priorities.begin(), parent->priorities.end());
      // std::cout<<"Max priorities : "<<max_init<<"\n";
      for (size_t i = 0; i < N; ++i) {
        float cur_dist = D.get(i, C[i]);
        if (cur_dist != 0) {
          priorities[i] = parent->priorities[i];
          // double z_d = (cur_dist-average_dist)/(std_dist+0.0001); z_d = std::min(z_d, 1.0); z_d = std::max(z_d, -1.0);z_d = (z_d+1)/2;
          double z_c = (D.get_conf(i)-average_conf)/(std_conf+0.0001); z_c = std::min(z_c, 1.0); z_c = std::max(z_c, -1.0);z_c = (z_c+1)/2;
          // double z_d = cur_dist/(max_dist+1);
          // double z_c =D.get_conf(i)/(max_conf+1);
          double z_n = (5-Planner::option[i])/5;
          
          double z_d = priorities[i]/max_init;
          
          CP[i] = float(z_d*parent->alpha+z_c*parent->beta+z_n*parent->gamma)+dist(gen);
          // CP[i] = z_d + dist(gen);
        } 
        else {  // at goal
          priorities[i] = parent->priorities[i] - (int)parent->priorities[i];
          CP[i] = priorities[i];
        }
      }
    }

    
    
    // std::iota(order.begin(), order.end(), 0);
    // std::sort(order.begin(), order.end(),
    //             [&](int i, int j) { return priorities[i] > priorities[j]; });

    // std::cout<<"******* Ordering by priorities ***********\n";
    // for (int i = 0; i<N; i++){
    //   std::cout<<order[i]<<"->";
    // }
    // std::cout<<"\n";


    // set order
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
                [&](int i, int j) { return CP[i] > CP[j]; });

    // std::cout<<"******* Ordering by CP ***********\n";
    // for (int i = 0; i<N; i++){
    //   std::cout<<order[i]<<"->";
    // }
    // std::cout<<"\n";

  }

    

}

Node::~Node()
{
  while (!search_tree.empty()) {
    delete search_tree.front();
    search_tree.pop();
  }
}


std::vector<int> Planner::option = {};
Planner::Planner(const Instance* _ins, const Deadline* _deadline,
                 std::mt19937* _MT, int _verbose)
    : ins(_ins),
      deadline(_deadline),
      MT(_MT),
      verbose(_verbose),
      N(ins->N),
      V_size(ins->G.size()),
      D(DistTable(ins)),
      C_next(Candidates(N, std::array<Vertex*, 5>())),
      tie_breakers(std::vector<float>(V_size, 0)),
      A(Agents(N, nullptr)),
      occupied_now(Agents(V_size, nullptr)),
      occupied_next(Agents(V_size, nullptr))
{
}

Solution Planner::solve()
{
  info(1, verbose, "elapsed:", elapsed_ms(deadline), "ms\tstart search");
  

  // setup agents
  for (auto i = 0; i < N; ++i) {
    for (auto i = 0; i < N; ++i) option.push_back(0);
    A[i] = new Agent(i);
  }

  // setup option table


  // setup search queues
  std::stack<Node*> OPEN;
  std::unordered_map<Config, Node*, ConfigHasher> CLOSED;
  std::vector<Constraint*> GC;  // garbage collection of constraints

  // insert initial node
  auto S = new Node(ins->starts, D, ins->h);
  OPEN.push(S);
  CLOSED[S->C] = S;

  // depth first search
  int loop_cnt = 0;
  std::vector<Config> solution;

  while (!OPEN.empty() && !is_expired(deadline)) {
    loop_cnt += 1;

    // do not pop here!
    S = OPEN.top();

    // check goal condition
    if (is_same_config(S->C, ins->goals)) {
      // backtrack
      while (S != nullptr) {
        solution.push_back(S->C);
        S = S->parent;
      }
      std::reverse(solution.begin(), solution.end());

      std::cout << "HNode: " << Node::HNODE_CNT << ", ";
      std::cout << "LNode: " << Constraint::LNODE_CNT << ", ";

      break;
    }

    // low-level search end
    if (S->search_tree.empty()) {
      OPEN.pop();
      continue;
    }

    // create successors at the low-level search
    auto M = S->search_tree.front();
    GC.push_back(M);
    S->search_tree.pop();
    if (M->depth < N) {
      auto i = S->order[M->depth];
      auto C = S->C[i]->neighbor;
      C.push_back(S->C[i]);
      int opt = C.size();
      // Planner::option[i] = C.size();
      
      if (MT != nullptr) std::shuffle(C.begin(), C.end(), *MT);  // randomize
      for (auto u : C) {
        
        S->search_tree.push(new Constraint(M, i, u));

      }
    }

    // create successors at the high-level search
    if (!get_new_config(S, M)) continue;

    // create new configuration
    auto C = Config(N, nullptr);
    for (auto a : A) C[a->id] = a->v_next;

    // check explored list
    auto iter = CLOSED.find(C);
    if (iter != CLOSED.end()) {
      OPEN.push(iter->second);
      continue;
    }

    // insert new search node
    auto S_new = new Node(C, D, ins->h, S);
    OPEN.push(S_new);
    CLOSED[S_new->C] = S_new;
  }

  info(1, verbose, "elapsed:", elapsed_ms(deadline), "ms\t",
       solution.empty() ? (OPEN.empty() ? "no solution" : "failed")
                        : "solution found",
       "\tloop_itr:", loop_cnt, "\texplored:", CLOSED.size());
  // memory management
  for (auto a : A) delete a;
  for (auto M : GC) delete M;
  for (auto p : CLOSED) delete p.second;

  return solution;
}

bool Planner::get_new_config(Node* S, Constraint* M)
{
  // setup cache
  for (auto a : A) {
    // clear previous cache
    if (a->v_now != nullptr && occupied_now[a->v_now->id] == a) {
      occupied_now[a->v_now->id] = nullptr;
    }
    if (a->v_next != nullptr) {
      occupied_next[a->v_next->id] = nullptr;
      a->v_next = nullptr;
    }

    // set occupied now
    a->v_now = S->C[a->id];
    occupied_now[a->v_now->id] = a;
  }

  // add constraints
  for (auto k = 0; k < M->depth; ++k) {
    const auto i = M->who[k];        // agent
    const auto l = M->where[k]->id;  // loc

    // check vertex collision
    if (occupied_next[l] != nullptr) return false;
    // check swap collision
    auto l_pre = S->C[i]->id;
    if (occupied_next[l_pre] != nullptr && occupied_now[l] != nullptr &&
        occupied_next[l_pre]->id == occupied_now[l]->id)
      return false;

    // set occupied_next
    A[i]->v_next = M->where[k];
    occupied_next[l] = A[i];
  }

  // perform PIBT
  for (auto k : S->order) {
    auto a = A[k];
    if (a->v_next == nullptr && !funcPIBT(a)) return false;  // planning failure
  }
  return true;
}

bool Planner::funcPIBT(Agent* ai)
{
  const auto i = ai->id;
  const auto K = ai->v_now->neighbor.size();
  Planner::option[i] = K;

  // get candidates for next locations
  for (size_t k = 0; k < K; ++k) {
    auto u = ai->v_now->neighbor[k];
    C_next[i][k] = u;
    if (MT != nullptr)
      tie_breakers[u->id] = get_random_float(MT);  // set tie-breaker
  }
  C_next[i][K] = ai->v_now;

  // sort, note: K + 1 is sufficient
  std::sort(C_next[i].begin(), C_next[i].begin() + K + 1,
            [&](Vertex* const v, Vertex* const u) {
              return D.get(i, v) + tie_breakers[v->id] <
                     D.get(i, u) + tie_breakers[u->id];
            });

  for (size_t k = 0; k < K + 1; ++k) {
    auto u = C_next[i][k];

    // avoid vertex conflicts
    if (occupied_next[u->id] != nullptr) {

      // std::cout<<"occupied_next: "<<occupied_next[u->id]->id<<"\n";

      auto& ak = occupied_next[u->id];
      D.update_conf(ai->id, ak->id);
      continue;
    }

    auto& ak = occupied_now[u->id];

    // avoid swap conflicts
    if (ak != nullptr && ak->v_next == ai->v_now) {

      D.update_conf(ai->id, ak->id);
      D.update_conf(ak->id, ai->id);
      
      continue;
    }

    // reserve next location
    occupied_next[u->id] = ai;
    ai->v_next = u;

    // empty or stay
    if (ak == nullptr || u == ai->v_now) return true;

    // priority inheritance
    if (ak->v_next == nullptr && !funcPIBT(ak)) continue;

    // success to plan next one step
    return true;
  }

  // failed to secure node
  occupied_next[ai->v_now->id] = ai;
  ai->v_next = ai->v_now;
  return false;
}

Solution solve(const Instance& ins, const int verbose, const Deadline* deadline,
               std::mt19937* MT)
{
  info(1, verbose, "elapsed:", elapsed_ms(deadline), "ms\tpre-processing");
  auto planner = Planner(&ins, deadline, MT, verbose);
  return planner.solve();
}
