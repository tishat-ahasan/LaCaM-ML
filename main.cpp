#include <argparse/argparse.hpp>
#include <lacam.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>






int main(int argc, char* argv[])
{
  // pwd = LaCam-ML



  // torch::jit::script::Module net = torch::jit::load("./Data/models/traced_model.pt");
  // std::vector<float> cpp_vector = {0.11, 0.15, 0.94, 0.06, 21.0, 1.0, 4.7, 5.76, 77.0, 0.0, 12.91, 14.65, 0.0, 0.0, 8.0, 59.0, 73.0, 140.0};
  // torch::Tensor x = torch::from_blob(cpp_vector.data(), {cpp_vector.size()}, torch::kFloat32);
  // x = x.unsqueeze(0);

  // std::vector<torch::jit::IValue> input;
  // input.push_back(x);
  // auto out = net.forward(input);
  // torch::Tensor out_tensor = out.toTensor();
  // std::cout << out_tensor[0][0].item<float>();
  // std::cout << typeid(out).name()<<"\n";




  // arguments parser
  argparse::ArgumentParser program("lacam", "0.1.0");
  program.add_argument("-m", "--map").help("map file").required();
  program.add_argument("-i", "--scen")
      .help("scenario file")
      .default_value(std::string(""));
  program.add_argument("-N", "--num").help("number of agents").required();
  program.add_argument("-s", "--seed")
      .help("seed")
      .default_value(std::string("0"));
  program.add_argument("-v", "--verbose")
      .help("verbose")
      .default_value(std::string("0"));
  program.add_argument("-t", "--time_limit_sec")
      .help("time limit sec")
      .default_value(std::string("10"));
  program.add_argument("-o", "--output")
      .help("output file")
      .default_value(std::string("./build/result.txt"));
  program.add_argument("-l", "--log_short")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("-h", "--heuristic")
      .default_value("distance")
      .help("heuristic");

  try {
    program.parse_known_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }

  // setup instance
  const auto verbose = std::stoi(program.get<std::string>("verbose"));
  const auto time_limit_sec =
      std::stoi(program.get<std::string>("time_limit_sec"));
  const auto scen_name = program.get<std::string>("scen");
  const auto seed = std::stoi(program.get<std::string>("seed"));
  auto MT = std::mt19937(seed);
  const auto map_name = program.get<std::string>("map");
  const auto output_name = program.get<std::string>("output");
  const auto log_short = program.get<bool>("log_short");
  const auto N = std::stoi(program.get<std::string>("num"));
  const auto heuristic = program.get<std::string>("heuristic");
  const auto ins = scen_name.size() > 0 ? Instance(scen_name, map_name, N, heuristic)
                                        : Instance(map_name, &MT, N, heuristic);
  
  // std::cout<<"heuristic: "<<heuristic<<"\n";
  if (!ins.is_valid(1)) return 1;

  // solve
  const auto deadline = Deadline(time_limit_sec * 1000);
  const auto solution = solve(ins, verbose - 1, &deadline, &MT);
  const auto comp_time_ms = deadline.elapsed_ms();

  // failure
  if (solution.empty()) info(1, verbose, "failed to solve");

  // check feasibility
  if (!is_feasible_solution(ins, solution, verbose)) {
    info(0, verbose, "invalid solution");
    return 1;
  }

  // post processing
  print_stats(verbose, ins, solution, comp_time_ms);
  make_log(ins, solution, output_name, comp_time_ms, map_name, seed, log_short);
  return 0;
}
