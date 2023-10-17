#include "../include/planner.hpp"
#include <cmath>

#include<torch/torch.h>


uint Constraint::LNODE_CNT = 0;
// torch::jit::script::Module net = torch::jit::load("./Data/models/traced_model.pt");

Constraint::Constraint() : who(std::vector<int>()), where(Vertices()), depth(0)
{
}

Constraint::Constraint(Constraint* parent, int i, Vertex* v)
    : who(parent->who), where(parent->where), depth(parent->depth + 1)
    
{
  ++LNODE_CNT;
  who.push_back(i);
  where.push_back(v);
  std::cout<<"Ekbar\n";
}

Constraint::~Constraint(){};

uint Node::HNODE_CNT = 0;
Node::Node(Config _C, DistTable& D, const std::string& _h, Node* _parent)
    : C(_C),
      parent(_parent),
      priorities(C.size(), 0),
      order(C.size(), 0),
      search_tree(std::queue<Constraint*>()),
      h(_h),
      net(torch::jit::load("./Data/models/traced_model.pt"))
{
  ++HNODE_CNT;
  search_tree.push(new Constraint());
  const auto N = C.size();

  bool is_print = false;

   // try {
   //      myModule = torch::jit::load("./Data/models/traced_model.pt");
   //  } catch (const c10::Error& e) {
   //      std::cerr << "Error loading the script: " << e.what() << std::endl;
   //      return -1;
   //  }

  std::vector<float> cpp_vector = {0.11, 0.15, 0.94, 0.06, 21.0, 1.0, 4.7, 5.76, 77.0, 0.0, 12.91, 14.65, 0.0, 0.0, 8.0, 59.0, 73.0, 140.0};
  torch::Tensor x = torch::from_blob(cpp_vector.data(), {cpp_vector.size()}, torch::kFloat32);
  x = x.unsqueeze(0);
  std::cout<<x<<"\n";
  std::vector<torch::jit::IValue> input;
  input.push_back(x);
  auto out = net.forward(input);
  torch::Tensor out_tensor = out.toTensor();
  std::cout << out_tensor[0][0].item<float>();
  std::cout << typeid(out).name()<<"\n";
  // ************************************************ Distance ******************************************
                // Percentages of agent at goal and non-goal position 2
                // Normalised (wrt nodes) Manhattan Distance (min, max, avg, std) 4
                // Normalised (wrt no of agents) Manhattan Distance (min, max, avg, std) 4
                // % of agents below or above avg distance 2
                // Normalised (wrt nodes) no of conflicts of each agent (min, max, avg, std) 4
                // Normalised (wrt no of agents) no of conflicts of each agent (min, max, avg, std) 4
                // % of agent below or above avg conflict 2
                // Option for next node selection (0,1,2,3,4) 

  //  csv order
  // % goal,% non-goal,
  // dis_max,dis_min,dis_avg,dis_std 
  // conf_max,conf_min,conf_avg,conf_std
  // neighbour 0, 1, 2, 3, 4

  // agents at goal or non goal
  float a_g = N;
  float a_ng = 0;
  std::vector<double> man_dist;
  man_dist.push_back(1);

  for (size_t i = 0; i < N; ++i) {
    float dist = D.get(i, C[i]);
    
    if ( dist!= 0) {  //at_goal
      man_dist.push_back(dist);
      a_ng++;
    }
  }
  // agents at goal or non goal
  a_g -= a_ng;
  a_g /= N;
  a_ng /= N;

  

  // Manhattan Distance
  double max_dist = *std::max_element(man_dist.begin(), man_dist.end());
  double min_dist = *std::min_element(man_dist.begin(), man_dist.end());
  double average_dist = std::accumulate(man_dist.begin(), man_dist.end(), 0.0) / man_dist.size();
  double sumOfSquares = 0.0;
  for (const double value : man_dist) {
      double diff = value - average_dist;
      sumOfSquares += diff * diff;
  }
  double std_dist = std::sqrt(sumOfSquares / man_dist.size());



  // conflicts

  double max_conf = *std::max_element(Planner::conflict.begin(), Planner::conflict.end());
  double min_conf = *std::min_element(Planner::conflict.begin(), Planner::conflict.end());
  double average_conf = std::accumulate(Planner::conflict.begin(), Planner::conflict.end(), 0.0) / N;
  sumOfSquares = 0.0;
  for (const double value : Planner::conflict) {
      double diff = value - average_conf;
      sumOfSquares += diff * diff;
  }
  double std_conf = std::sqrt(sumOfSquares / N);

  // neighbour
  std::vector<float> neighbour(5, 0.0);
  for (size_t i = 0; i < N; ++i) {
    neighbour[Planner::option[i]]++;
  }

  // Print Stats....
  std::cout<<a_g<<","<<a_ng<<",";
  std::cout<<max_dist<<","<<min_dist<<","<<average_dist<<","<<std_dist<<",";
  std::cout<<max_conf<<","<<min_conf<<","<<average_conf<<","<<std_conf<<",";
  for (int i = 0; i<4; ++i){
    std::cout<<neighbour[i]<<",";
  }
  std::cout<<neighbour[4]<<"\n";


  // set priorities
  
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
              [&](int i, int j) { return priorities[i] > priorities[j]; });
  }

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
              [&](int i, int j) { return priorities[i] > priorities[j]; });

  }

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
          priorities[i] = Planner::conflict[i];
        } else {
          priorities[i] = parent->priorities[i] - (int)parent->priorities[i];
        }
      }
    }
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int i, int j) { return priorities[i] > priorities[j]; });

  }
  // ************************************************ Neighbour ******************************************

  // set order using option
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
              [&](int i, int j) { return priorities[i] > priorities[j]; });

  }
  // print check
  // for (int element : Planner::option) {
  //       std::cout << element << " ";
  //   }
  //   std::cout << std::endl;

}

Node::~Node()
{
  while (!search_tree.empty()) {
    delete search_tree.front();
    search_tree.pop();
  }
}


std::vector<int> Planner::conflict = {};
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
  for (auto i = 0; i < N; ++i) A[i] = new Agent(i);

  // setup conflict table
  for (auto i = 0; i < N; ++i) conflict.push_back(0);
  // setup option table
  for (auto i = 0; i < N; ++i) option.push_back(0);

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
      Planner::conflict[ai->id]++;
      continue;
    }

    auto& ak = occupied_now[u->id];

    // avoid swap conflicts
    if (ak != nullptr && ak->v_next == ai->v_now) {

      Planner::conflict[ai->id]++;
      Planner::conflict[ak->id]++;
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
