#include "./AppCom/AppCom.hpp"
#include "./AppDec/AppDec.hpp"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

typedef std::function<int(int, char **)> command_func_t;

int ShowHelp(
    const std::vector<std::pair<std::string, command_func_t>> &commands)
{

  std::cout << "Usage:" << std::endl;
  std::cout << "  ./<bin> [command] [options]" << std::endl
            << std::endl;

  std::cout << "Documentation:" << std::endl;
  std::cout << "  ./doc/" << std::endl
            << std::endl;

  std::cout << "Example usage:" << std::endl;
  std::cout << "  <bin> dec [dateset path] #decompress" << std::endl;
  std::cout << "  <bin> com #compress" << std::endl;
  std::cout << "  ..." << std::endl
            << std::endl;

  std::cout << "Available commands:" << std::endl;
  std::cout << "  help" << std::endl;
  for (const auto &command : commands)
  {
    std::cout << "  " << command.first << std::endl;
  }
  std::cout << std::endl;

  return 0;
}

int main(int argc, char **argv)
{
  ck(cuInit(0));

  std::vector<std::pair<std::string, command_func_t>> commands;

  commands.emplace_back("dec", &dec_pool);      //通过线程池解压hevc文件
  commands.emplace_back("com", &compresse_bag); //Compress images in bags. The directory tree is fixed in this setting.
  commands.emplace_back("dir", &compresse_dir); //压缩指定目录中图片

  if (argc == 1)
  {
    return ShowHelp(commands);
  }

  const std::string command = argv[1];
  if (command == "help" || command == "-h" || command == "--help")
  {
    return ShowHelp(commands);
  }
  else
  {
    command_func_t matched_command_func = nullptr;
    for (const auto &command_func : commands)
    {
      if (command == command_func.first)
      {
        matched_command_func = command_func.second;
        break;
      }
    }
    if (matched_command_func == nullptr)
    {
      std::cerr << "ERROR: Command `%s` not recognized. To list the "
                   "available commands, run `colmap help`."
                << command.c_str()
                << std::endl;
      return EXIT_FAILURE;
    }
    else
    {
      int command_argc = argc - 2;
      char **command_argv = &argv[2];
      // command_argv[0] = argv[0];
      return matched_command_func(command_argc, command_argv);
    }
  }
  return ShowHelp(commands);

  return 0;
}
