# import yaml

# class Yaml_Config():

#     def __init__(self, config_name):

#         with open(f"../configs/{config_name}", 'r') as ymlfile:
#             cfg = yaml.load(ymlfile)

#             self.host = cfg["mysql"]["host"]
#             self.database = cfg["mysql"]["database"]
#             self.user = cfg["mysql"]["user"]
#             self.password = cfg["mysql"]["password"]


# ymlconf = Yaml_Config('config_cluster.yaml')

# print(ymlconf.host)
# print(ymlconf.database)

# from ruamel.yaml import YAML

from ruamel.yaml import YAML
import logging

class YParams():
  """ Yaml file parser """
  def __init__(self, yaml_filename, print_params=False):
    self._yaml_filename = yaml_filename
    self.params = {}

    with open(yaml_filename) as _file:

      for key, val in YAML().load(_file).items():
        if val =='None': val = None

        self.params[key] = val
        self.__setattr__(key, val)

    if print_params:
      self.log()

  def __getitem__(self, key):
    return self.params[key]

  def __setitem__(self, key, val):
    self.params[key] = val

  def log(self):
    logging.info("------------------ Configuration ------------------")
    logging.info("Configuration file: "+str(self._yaml_filename))
    for key, val in self.params.items():
      logging.info(str(key) + ' ' + str(val))
    logging.info("---------------------------------------------------")

  def update(self, new_params):
    self.params.update(new_params)
    for key, val in new_params.items():
      self.__setattr__(key, val)


if __name__ == '__main__':
    yaml_filename = '../configs/default_config.yaml'
    test_yaml = YParams(yaml_filename)
    print (test_yaml.params)
